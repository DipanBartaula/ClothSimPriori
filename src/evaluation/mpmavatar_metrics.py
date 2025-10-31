"""
Metrics inspired by cloth evaluation in papers like MPMAvatar.

Functions here are written to be flexible: they accept point clouds,
mesh vertex arrays and face indices, normals, and optional contact masks
or signed distances. They aim to reproduce the kinds of metrics typically
reported in differentiable cloth works: Chamfer/Hausdorff distance between
predicted and ground-truth geometry, normal consistency, contact precision/
recall and penetration statistics.

Notes:
- For point-to-mesh distances we compute unsigned distances from points to
  triangles using a vectorized closest-point-on-triangle calculation.
- Penetration statistics require either signed distances for cloth points
  to the body mesh (negative = inside) or else an external SDF/occupancy
  that you provide per point.
"""

from typing import Optional, Tuple
import torch
import torch.nn.functional as F


def hausdorff_distance(pc1: torch.Tensor, pc2: torch.Tensor) -> torch.Tensor:
    """Symmetric Hausdorff distance between two point clouds.

    Args:
        pc1: (B,N,3) or (N,3)
        pc2: (B,M,3) or (M,3)

    Returns:
        scalar or (B,) tensor with Hausdorff distances.
    """
    if pc1.dim() == 2:
        pc1 = pc1.unsqueeze(0)
    if pc2.dim() == 2:
        pc2 = pc2.unsqueeze(0)

    # pairwise squared distances
    pc1_sq = (pc1 ** 2).sum(dim=2, keepdim=True)
    pc2_sq = (pc2 ** 2).sum(dim=2, keepdim=True)
    dists = pc1_sq + pc2_sq.transpose(1, 2) - 2 * (pc1 @ pc2.transpose(1, 2))
    dists = torch.clamp(dists, min=0.0)

    # directed distances
    d1 = torch.sqrt(dists.min(dim=2)[0])  # (B,N) nearest dist from pc1->pc2
    d2 = torch.sqrt(dists.min(dim=1)[0])  # (B,M)

    # Hausdorff: max of mean or max? commonly use max of nearest distances
    hd = torch.max(d1.max(dim=1)[0], d2.max(dim=1)[0])
    return hd if hd.numel() > 1 else hd.squeeze(0)


def _closest_point_on_triangle(p: torch.Tensor, a: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    """
    Vectorized computation of closest point on triangle ABC for each point P.

    Shapes:
      p: (..., 3)
      a,b,c: (..., 3)

    Returns:
      closest point Q with shape (...,3)
    """
    # Based on standard algorithm (see Real-Time Collision Detection)
    ab = b - a
    ac = c - a
    ap = p - a

    d1 = (ab * ap).sum(dim=-1)
    d2 = (ac * ap).sum(dim=-1)

    # Check if P in vertex region outside A
    cond_a = (d1 <= 0) & (d2 <= 0)
    if cond_a.any():
        q_a = a

    # Check vertex region outside B
    bp = p - b
    d3 = (ab * bp).sum(dim=-1)
    d4 = (ac * bp).sum(dim=-1)
    cond_b = (d3 >= 0) & (d4 <= d3)

    # Check edge region AB
    vc = d1 * d4 - d3 * d2
    cond_ab = (vc <= 0) & (d1 >= 0) & (d3 <= 0)

    # Check vertex region outside C
    cp = p - c
    d5 = (ab * cp).sum(dim=-1)
    d6 = (ac * cp).sum(dim=-1)
    cond_c = (d6 >= 0) & (d5 <= d6)

    # Check edge region AC
    vb = d5 * d2 - d1 * d6
    cond_ac = (vb <= 0) & (d2 >= 0) & (d6 <= 0)

    # Check edge region BC
    va = d3 * d6 - d5 * d4
    cond_bc = (va <= 0) & ((d4 - d3) >= 0) & ((d5 - d6) >= 0)

    # Otherwise P inside face region. Project onto plane and return barycentric combination
    # Compute projection onto plane
    denom = (ab.cross(ac)).norm(dim=-1) ** 2
    # Avoid division issues
    denom = torch.clamp(denom, min=1e-12)

    # For simplicity, we'll compute the true closest points by checking candidates:
    # projection onto plane and closest points on edges/vertices, then pick min distance.
    # Candidate 1: projection onto plane
    n = torch.cross(ab, ac)
    n_norm = n / (n.norm(dim=-1, keepdim=True) + 1e-12)
    dist_plane = (ap * n_norm).sum(dim=-1)
    proj = p - dist_plane.unsqueeze(-1) * n_norm

    # Candidate 2/3/4: closest points on edges AB, BC, AC
    def closest_point_on_segment(p, s0, s1):
        v = s1 - s0
        t = ((p - s0) * v).sum(dim=-1) / (v * v).sum(dim=-1)
        t = t.clamp(0.0, 1.0)
        return s0 + t.unsqueeze(-1) * v

    q_ab = closest_point_on_segment(p, a, b)
    q_ac = closest_point_on_segment(p, a, c)
    q_bc = closest_point_on_segment(p, b, c)

    candidates = torch.stack([proj, q_ab, q_ac, q_bc, a, b, c], dim=0)  # (7,...,3)
    # compute distances
    diffs = candidates - p.unsqueeze(0)
    d2s = (diffs ** 2).sum(dim=-1)  # (7,...)
    min_idx = d2s.argmin(dim=0)

    # pick corresponding candidate per point
    # gather along first dim
    # reshape for gather: bring first dim to last for indexing
    # Use advanced indexing
    idx = min_idx
    # Build output by selecting per-point
    # candidates shape (7, ..., 3). We want result shape (...,3)
    res = torch.stack([candidates[i].reshape(-1, 3) for i in range(candidates.shape[0])], dim=0)
    res = res[idx.reshape(-1)].reshape(p.shape)
    return res


def point_to_mesh_distance(points: torch.Tensor, verts: torch.Tensor, faces: torch.Tensor) -> torch.Tensor:
    """Compute unsigned closest distance from each point to the surface defined by verts/faces.

    Args:
        points: (P,3) or (B,P,3)
        verts: (V,3)
        faces: (F,3) index tensor into verts

    Returns:
        distances: (P,) or (B,P) tensor of unsigned distances.
    """
    # For simplicity, support unbatched points (P,3) and single mesh
    if points.dim() == 3 and points.shape[0] == 1:
        points = points.squeeze(0)

    assert points.dim() == 2 and verts.dim() == 2 and faces.dim() == 2

    device = points.device
    P = points.shape[0]
    F = faces.shape[0]
    tris = verts[faces]  # (F,3,3)

    # Expand to compute (P, F, 3)
    p_exp = points.unsqueeze(1).expand(-1, F, -1)
    a = tris[:, 0, :].unsqueeze(0).expand(P, -1, -1)
    b = tris[:, 1, :].unsqueeze(0).expand(P, -1, -1)
    c = tris[:, 2, :].unsqueeze(0).expand(P, -1, -1)

    p_flat = p_exp.reshape(-1, 3)
    a_flat = a.reshape(-1, 3)
    b_flat = b.reshape(-1, 3)
    c_flat = c.reshape(-1, 3)

    # compute closest points per (P*F) and then reduce
    q_flat = _closest_point_on_triangle(p_flat, a_flat, b_flat, c_flat)
    d2_flat = ((p_flat - q_flat) ** 2).sum(dim=-1)
    d2 = d2_flat.reshape(P, F)
    mins, _ = d2.min(dim=1)
    return torch.sqrt(mins)


def normal_consistency(pred_verts: torch.Tensor, pred_normals: torch.Tensor, gt_verts: torch.Tensor, gt_normals: torch.Tensor) -> torch.Tensor:
    """Compute mean absolute dot product between matched normals.

    Matches are computed by nearest neighbor between vertices.
    Returns mean(|dot(n_pred, n_gt)|) in [0,1], higher is better.
    """
    if pred_verts.dim() == 2:
        pred_verts = pred_verts.unsqueeze(0)
        pred_normals = pred_normals.unsqueeze(0)
        gt_verts = gt_verts.unsqueeze(0)
        gt_normals = gt_normals.unsqueeze(0)

    # pairwise distances
    pv_sq = (pred_verts ** 2).sum(dim=2, keepdim=True)
    gv_sq = (gt_verts ** 2).sum(dim=2, keepdim=True)
    dists = pv_sq + gv_sq.transpose(1, 2) - 2 * (pred_verts @ gt_verts.transpose(1, 2))
    nn_idx = dists.argmin(dim=2)  # (B, Npred)

    B, Npred, _ = pred_verts.shape
    dots = []
    for b in range(B):
        matched_gt_normals = gt_normals[b][nn_idx[b]]  # (Npred,3)
        n_pred = pred_normals[b]
        dot = (n_pred * matched_gt_normals).sum(dim=1).abs()
        dots.append(dot.mean())
    out = torch.stack(dots)
    return out if out.numel() > 1 else out.squeeze(0)


def contact_precision_recall(sim_contact_mask: torch.Tensor, gt_contact_mask: torch.Tensor) -> Tuple[float, float]:
    """Compute precision and recall for contact predictions.

    Args:
        sim_contact_mask: boolean tensor (T,N) or (N,) indicating predicted contact.
        gt_contact_mask: boolean tensor same shape indicating ground-truth contact.

    Returns:
        (precision, recall) as floats or tensors per batch.
    """
    assert sim_contact_mask.shape == gt_contact_mask.shape
    sim = sim_contact_mask.bool()
    gt = gt_contact_mask.bool()

    tp = (sim & gt).sum().item()
    pred_pos = sim.sum().item()
    gt_pos = gt.sum().item()

    precision = tp / pred_pos if pred_pos > 0 else 0.0
    recall = tp / gt_pos if gt_pos > 0 else 0.0
    return precision, recall


def penetration_statistics(signed_distances: torch.Tensor) -> dict:
    """
    Given signed distances from cloth points to body mesh (negative = penetration),
    compute simple penetration statistics.

    Args:
        signed_distances: (P,) or (B,P)

    Returns:
        dict with keys: mean_penetration, max_penetration, penetration_rate
    """
    if signed_distances.dim() == 1:
        sd = signed_distances.unsqueeze(0)
    else:
        sd = signed_distances

    # penetration depths are negative signed distances clipped
    pen = torch.clamp(-sd, min=0.0)
    mean_pen = pen.mean(dim=1)
    max_pen = pen.max(dim=1)[0]
    rate = (pen > 0).float().mean(dim=1)

    out = {
        "mean_penetration": mean_pen if mean_pen.numel() > 1 else mean_pen.squeeze(0),
        "max_penetration": max_pen if max_pen.numel() > 1 else max_pen.squeeze(0),
        "penetration_rate": rate if rate.numel() > 1 else rate.squeeze(0),
    }
    return out


__all__ = [
    "hausdorff_distance",
    "point_to_mesh_distance",
    "normal_consistency",
    "contact_precision_recall",
    "penetration_statistics",
]

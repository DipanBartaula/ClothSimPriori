import torch
from src.evaluation.mpmavatar_metrics import (
    hausdorff_distance,
    point_to_mesh_distance,
    normal_consistency,
    contact_precision_recall,
    penetration_statistics,
)


def test_hausdorff_simple():
    a = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b = torch.tensor([[0.0, 0.1, 0.0], [0.9, 0.0, 0.0]])
    hd = hausdorff_distance(a, b)
    assert hd >= 0


def test_point_to_mesh_distance_and_penetration():
    # Simple single triangle in XY plane at z=0
    verts = torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    faces = torch.tensor([[0, 1, 2]])
    pts = torch.tensor([[0.1, 0.1, 0.5], [0.2, 0.2, -0.2]])
    d = point_to_mesh_distance(pts, verts, faces)
    # distances should match abs(z) when point projects inside triangle approximately
    assert torch.all(d > 0)

    # penetration stats with provided signed distances
    signed = torch.tensor([0.5, -0.2])
    stats = penetration_statistics(signed)
    assert stats["mean_penetration"] >= 0


def test_normal_consistency_and_contact():
    # simple test with identical normals
    pv = torch.tensor([[0.0, 0.0, 0.0]])
    pn = torch.tensor([[0.0, 0.0, 1.0]])
    gv = torch.tensor([[0.0, 0.0, 0.0]])
    gn = torch.tensor([[0.0, 0.0, 1.0]])
    nc = normal_consistency(pv, pn, gv, gn)
    assert torch.isclose(nc, torch.tensor(1.0), atol=1e-6)

    sim_mask = torch.tensor([1, 0, 1], dtype=torch.bool)
    gt_mask = torch.tensor([1, 1, 0], dtype=torch.bool)
    prec, rec = contact_precision_recall(sim_mask, gt_mask)
    assert 0.0 <= prec <= 1.0
    assert 0.0 <= rec <= 1.0

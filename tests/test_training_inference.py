import torch
from src.models.physical_parameters import ClothPhysicalParameters

# We just test that the parameter module works
def test_physical_parameters():
    params = ClothPhysicalParameters()
    
    # Test default values
    assert params.stiffness.item() > 0
    assert params.damping.item() > 0
    assert params.friction.item() > 0 and params.friction.item() < 1
    assert params.mass_density.item() > 0
    
    # Test optimization
    optimizer = torch.optim.SGD(params.parameters(), lr=1.0)
    
    # Get initial value
    initial_stiffness = params.stiffness.item()
    
    # Mock loss and backward
    loss = params.stiffness * 2.0
    loss.backward()
    optimizer.step()
    
    # Check that value changed
    new_stiffness = params.stiffness.item()
    assert new_stiffness != initial_stiffness
    print("PhysicalParameters test passed.")

# A full training loop test is complex, but we can test the param module.
# The `train.py` and `inference.py` are tested by running them.
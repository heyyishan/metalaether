"""Placeholder for model-format validation; run after building the Python extension."""
import aether, os, tempfile
c=aether.Config(); c.dim=32; c.hidden_dim=64; c.n_layers=1; c.n_heads=4; c.max_seq_len=32
m=aether.Model.random_for_demo(c)
with tempfile.NamedTemporaryFile(suffix='.aether') as f:
    m.save(f.name); m2=aether.Model.load(f.name)
    assert m2.config.dim == 32 and m2.config.n_layers == 1
print('model format round-trip: PASS')

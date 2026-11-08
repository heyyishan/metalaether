import aether
c=aether.Config(); c.dim=128; c.hidden_dim=384; c.n_layers=2; c.n_heads=4; c.max_seq_len=256
m=aether.Model.random_for_demo(c); m.save('demo.aether')
print('Wrote demo.aether (random weights; smoke test only, not a pretrained language model).')

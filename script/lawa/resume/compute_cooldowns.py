
## OGBG / CONFORMER, step_hint=80_000, save_evry=64, eval_every=256
80_000 / 4  # 20_000

# cooldown should end at 20k, 40k, 60k
# cooldown legth: ~10% = 8k

# start at:
(20_000 - 8000) // 64 * 64  # 11968
(40_000 - 8000) // 64 * 64  # 32000
(60_000 - 8000) // 64 * 64  # 51968


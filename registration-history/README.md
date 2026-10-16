# Registration history

The paper cites its registrations by the commit that adds them (Appendix D): v0 `f469617`, v1
`2051086`, v2 `525f7ab`, v3 `a4b5474`, and v4 `72261c0` (the round-3 blocks). Those commits live in the
authors' working repository, whose files also hold internal infrastructure details. This directory
publishes the part of that history needed to check the order of the registrations, without the rest
of the working tree:

- `objects/*.commit`: the raw commit objects of the whole history up to the head named in `index.json`
  (message, author, dates, tree and parent hashes), except the first commit, which is given by its hash
  only because its message names an internal host.
- `objects/*.tree`, `objects/*.blob`: for each registration, the tree objects on the path to its file
  and the file itself as committed.
- `verify.py`: recomputes every object hash, follows the parent chain, places each registration file in
  its commit, and checks which run manifests in `runs/` were produced at a commit that descends from
  each registration.

Run `python registration-history/verify.py` from the repository root.

What this establishes: the registration texts were committed in this order, before the commits that
produced the runs that descend from them, in a hash chain that cannot be altered without changing the
cited hashes. What it does not establish: wall-clock times. Git dates are set by the committer, so the
order of commits, not their dates, carries the claim. The head hash in `index.json` is the history as of
the release that added this directory.

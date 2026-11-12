# Historical platform validation, 2026-10-20

This predates the controller integration and practice package extraction.
See README.md for the subsequent relay results and current validation limits.

Project identity was verified through authenticated Sandbox and registry APIs.
The platform AccessKey can log into the primary registry. The Sandbox registry
uses separate credentials obtained through the platform's Harbor v2 account API;
the same AccessKey is not itself the Sandbox registry username/password.

A dedicated task image was pushed successfully and a dedicated Environment was
created with capacity 2 and prewarm 0. A GPU-local runner completed real Sandbox
creation/readiness, positive and negative command exit-code checks, binary file
upload/download, and acknowledged deletion. These checks used the existing
prepared humanevalfix task image, not a newly built GPU environment image.

Gateway reachability is blocked. The GPU-local HTTP probe responds with HTTP 200
on loopback. Sandbox connections to two GPU interface addresses timed out. The
Environment initially ignored egressEnabled during creation. A subsequent PATCH
requires networkPolicy; setting default deny with explicit GPU /32 allow rules
succeeded, but the connection still timed out. No training was launched.

One allocation timed out during SDK endpoint initialization. SDK cleanup was
followed by an explicit API check: the instance returned 404 and the Sandbox list
was empty. Subsequent controlled validation succeeded through command/files but
again failed Gateway reachability, then acknowledged instance deletion.

Next gate: platform networking must provide a Sandbox-reachable GPU listener or
resolve the current route/policy restriction. Do not claim RL_READY or training
acceptance until a live group passes the existing session and token audits.
Deployment-specific IDs, addresses and detailed probe evidence are maintained
outside the public source tree in the validation directory on shared storage.

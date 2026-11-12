"""Optional H-cluster deployment adapters; not imported by HarborRL core.

Modules in this package, each usable alone:

``environment``   OpenSandbox platform environment adapter (site Environment
                  IDs, mounts, command port) used by the sandbox runtime.
``transport``     bounded retries for platform endpoint discovery; allocation
                  POSTs are never retried.
``relay``         manager-side relay pump for sandboxes that may only reach the
                  platform proxy.
``relay_server``  the loopback mailbox the relay and a sandbox share.
``reverse_relay`` manager-initiated variant of the same relay.
``rjob``          ``gpu_target.provision`` implementation: submit and release a
                  GPU pod, print the machine record the core consumes.
``tunnel``        reverse ssh tunnel so a host the cluster cannot dial can still
                  serve gateway shard ports, plus the artifact push back.

Nothing here is imported at package import time; each adapter is loaded by the
component that needs it.
"""

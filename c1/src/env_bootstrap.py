#!/usr/bin/env python3
"""
Point the Hugging Face cache at a volume with room, before anything imports it.

The default cache lives under ~/.cache/huggingface on the root filesystem,
which on this machine has ~2 GB free at 98% full. The Phase 0 grid needs
`medium` (~1.5 GB) and `large-v3` (~3.1 GB); downloading them there would fill
the root filesystem and take the machine down with it, not just the experiment.

/mnt/F has ~20 GB free, so the cache goes there. HF_HOME is read by
huggingface_hub at import time, so this module must be imported BEFORE
huggingface_hub, faster_whisper or transformers — importing it later has no
effect and the cache silently stays on the full volume.

Override with SORA_HF_HOME to put the cache somewhere else.
"""
import os
import socket

DEFAULT_HF_HOME = "/mnt/F/SLIIT/Research/.hf_cache"

# Xet is huggingface_hub's chunked storage backend (the `hf-xet` package). On
# this network it hangs indefinitely rather than failing: no error, nothing
# written to disk, 0% CPU, and the HF_HUB_*_TIMEOUT settings do not trip. Two
# large-v3 fetches stalled this way, and it is the most likely cause of the
# minutes-long WhisperModel() load too.
#
# Plain HTTPS to the same files works fine — measured at ~640 KB/s with curl —
# so the classic download path is forced here. Set SORA_ALLOW_XET=1 to try Xet
# again if the network situation changes.
if not os.environ.get("SORA_ALLOW_XET"):
    os.environ.setdefault("HF_HUB_DISABLE_XET", "1")


def setup_hf_home(verbose=False):
    """
    Set HF_HOME unless the caller already chose one. Returns the path in use.

    Respects an existing HF_HOME so a teammate's own setup is never overridden.
    """
    if os.environ.get("HF_HOME"):
        if verbose:
            print(f"HF_HOME already set: {os.environ['HF_HOME']}")
        return os.environ["HF_HOME"]

    hf_home = os.environ.get("SORA_HF_HOME", DEFAULT_HF_HOME)
    os.makedirs(hf_home, exist_ok=True)
    os.environ["HF_HOME"] = hf_home
    if verbose:
        print(f"HF_HOME -> {hf_home}")
    return hf_home


def ipv6_is_routable(host="huggingface.co", timeout=3.0):
    """Can this machine actually open an IPv6 connection? Cheap, fails fast."""
    try:
        infos = socket.getaddrinfo(host, 443, socket.AF_INET6, socket.SOCK_STREAM)
    except OSError:
        return False
    for *_, addr in infos[:1]:
        sock = socket.socket(socket.AF_INET6, socket.SOCK_STREAM)
        sock.settimeout(timeout)
        try:
            sock.connect(addr)
            return True
        except OSError:
            return False
        finally:
            sock.close()
    return False


def force_ipv4(verbose=False):
    """
    Make AF_UNSPEC lookups return IPv4 only, when IPv6 is advertised but dead.

    This network resolves IPv6 addresses for huggingface.co but cannot route
    them: an IPv4 connect completes in 0.13s while IPv6 times out. getaddrinfo
    returns the IPv6 records first (RFC 6724), and urllib3 tries candidates
    serially, so `requests` blocks for minutes — well past its own timeout,
    because a connect that is stalled per-address is not one slow request but
    several. curl is unaffected: Happy Eyeballs races both families and takes
    IPv4 immediately, which is why curl worked while every Python client hung.

    That single fault explains all three stalls seen here — snapshot_download
    writing nothing, the direct requests download, and the minutes-long
    WhisperModel() load, which contacts the Hub on the same stack.

    Only AF_UNSPEC is redirected, so code explicitly asking for AF_INET6 still
    gets it. Set SORA_ALLOW_IPV6=1 to skip this entirely.
    """
    if os.environ.get("SORA_ALLOW_IPV6"):
        return False
    if getattr(socket, "_sora_ipv4_forced", False):
        return True
    if ipv6_is_routable():
        if verbose:
            print("IPv6 is routable; leaving resolution alone")
        return False

    original = socket.getaddrinfo

    def ipv4_only(host, port, family=0, type=0, proto=0, flags=0):
        if family == socket.AF_UNSPEC:
            family = socket.AF_INET
        return original(host, port, family, type, proto, flags)

    socket.getaddrinfo = ipv4_only
    socket._sora_ipv4_forced = True
    if verbose:
        print("IPv6 advertised but not routable -> forcing IPv4 for this process")
    return True


def free_gb(path):
    import shutil
    return shutil.disk_usage(path).free / 1e9


# Both run on import: callers only have to import this module early enough.
HF_HOME = setup_hf_home()
IPV4_FORCED = force_ipv4()


if __name__ == "__main__":
    print(f"HF_HOME     : {HF_HOME}")
    print(f"IPv4 forced : {IPV4_FORCED}")
    print(f"free there  : {free_gb(HF_HOME):.1f} GB")
    print(f"free on /    : {free_gb('/'):.1f} GB")
    hub = os.path.join(HF_HOME, "hub")
    if os.path.isdir(hub):
        for name in sorted(os.listdir(hub)):
            if name.startswith("models--"):
                print(f"  cached: {name}")

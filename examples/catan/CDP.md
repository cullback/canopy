# Chrome DevTools Protocol (CDP) Connection

Connect to a Chrome instance running colonist.io to read game state.

## Setup

### 1. Launch Chrome on Mac with debug port

Quit all Chrome processes first (check Activity Monitor), then:

```
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --remote-debugging-port=9222 \
  --user-data-dir=~/Desktop/chrome-cdp-profile
```

Verify: `curl http://localhost:9222/json` should list open tabs.

### 2. SSH tunnel from NixOS VM to Mac

Run on the **VM** (not the Mac):

```
ssh -N -L 9223:127.0.0.1:9222 cullback@192.168.64.1
```

- `192.168.64.1` is the Mac's UTM bridge IP
- Port 9223 on VM side (9222 may be occupied)

### 3. Verify from VM

```
curl http://localhost:9223/json/version
curl http://localhost:9223/json
```

## Gotchas

- `--remote-debugging-address=0.0.0.0` is ignored by modern Chrome — it only binds to localhost. SSH tunnel is required.
- SSH `-L` binds the port on the machine where you run ssh. Run it from the VM, not the Mac.
- Uses the UTM bridge network (`192.168.64.x`), not Tailscale.
- `--user-data-dir` creates a fresh profile. Omit to use your default profile (but must still quit Chrome first).

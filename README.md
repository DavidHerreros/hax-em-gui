# Hax Web GUI

A web interface for running the `hax-em` programs on a GPU server from any
browser — no command line required. Fill in a form, launch a job, and watch it
run, all from the same page.

## What you can do

- **Run any `hax-em` program** through an auto-generated form, with the same
  options you'd pass on the command line — no need to remember argument names.
- **Launch and monitor jobs**, with live logs streamed to your browser as they run.
- **View training metrics** with an embedded TensorBoard (the `display_metrics` tool).
- **Organise your work into projects**, each keeping its own runs and output folders.
- **Browse server files** to pick inputs and outputs without leaving the interface.

It runs on the GPU server; your browser can be anywhere. Because everything is
served from a single address, there is nothing to install in the browser — no
Node, no npm, no build step.

## Install

Install it into the same Conda environment that already has `hax-em`:

```bash
pip install hax-em-gui
```

This adds the `hax-gui` command and everything the interface needs.

Whenever a `hax-em` program's options change (for example after upgrading
`hax-em`), refresh the form definitions once:

```bash
hax-gui schemas
```

## Run it

```bash
hax-gui start                 # start in the background, on http://127.0.0.1:8000
hax-gui status                # is it running? where? pid + health
hax-gui restart               # relaunch (keeps the current host/port)
hax-gui stop                  # stop it
```

To reach the interface from other machines, bind it to a public address:

```bash
hax-gui start --host 0.0.0.0 --port 8051
```

Then open **http://&lt;server&gt;:&lt;port&gt;** in any browser. Add `--foreground`
to keep the server attached to your terminal instead of running in the background.

## Configuration

The server reads a few optional environment variables:

| Variable | Default | Purpose |
|---|---|---|
| `HAX_GUI_DATA_DIR` | `~/.hax_gui` | Where job logs and the run database are stored |
| `HAX_GUI_FILE_ROOTS` | _(unrestricted)_ | Colon-separated folders to **confine** the file browser to. Unset, it shows anything the server can read. Not a security boundary: job arguments accept any path either way |
| `HAX_GUI_BROWSE_HOME` | `$HOME` | Folder the file browser opens at |
| `HAX_GUI_TOKEN` | _(empty)_ | Shared access token. On a non-loopback bind one is **always** enforced: if unset, a token is generated once and kept in `<data dir>/token` (`hax-gui start` prints its location) |
| `HAX_GUI_ALLOWED_HOSTS` | _(empty)_ | Extra `Host` header values to accept (comma-separated), for reverse-proxy setups; `*` disables the check. Loopback binds accept only loopback hostnames (DNS-rebinding protection) |
| `HAX_PROJECT_MANAGER` | `hax_project_manager` | The `hax-em` launcher command to use |

## Remote use & security

When you expose the interface beyond `localhost`, an access token is enforced
automatically: `hax-gui start --host 0.0.0.0` generates one on first use (kept
in `<data dir>/token`) unless `HAX_GUI_TOKEN` is set. Put a reverse proxy (such
as Caddy or nginx) in front to terminate HTTPS, and set `HAX_GUI_ALLOWED_HOSTS`
to the public name(s) the proxy serves. The backend stays on the GPU server;
the browser can be anywhere.

On the default loopback bind the server stays open (no token) but accepts only
loopback `Host` headers, which blocks DNS-rebinding attacks from hostile web
pages. Set `HAX_GUI_TOKEN` anyway if other people can log into the machine.

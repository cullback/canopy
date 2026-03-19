set dotenv-load := true

# Display available recipes
default:
    just --list --unsorted

# Install dependencies and set up the development environment
bootstrap:
    cargo build

alias fmt := format

# Format code
format:
    dprint fmt
    cargo fmt
    fd -e nix | xargs -r nixfmt
    rg -l '[^\n]\z' --multiline | xargs -r sed -i -e '$a\\'

# Run linters and static analysis
check:
    dprint check
    cargo fmt --check
    cargo clippy -- -D warnings
    fd -e nix | xargs -r nixfmt --check
    ! rg -l '[^\n]\z' --multiline

# Run the test suite
test:
    cargo test

# Build the library
build:
    cargo build --release

# Run an example (e.g., just run pig)
run example *args:
    cargo run --release --example {{ example }} -- {{ args }}

# Sync project to a remote host (set VASTAI_HOST, VASTAI_USER, VASTAI_PORT in .env)
sync:
    rsync -avz --delete --filter=':- .gitignore' -e "ssh -p $VASTAI_PORT" . "$VASTAI_USER@$VASTAI_HOST:/workspace/canopy/"

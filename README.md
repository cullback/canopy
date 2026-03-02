# Template Rust

A starting point for Rust projects that constrains LLM-assisted development to a known-good structure.

## Setup Instructions

Rename the package to match your project:

1. Update `name` in `Cargo.toml`
2. Update `pname`, `description`, and `mainProgram` in `flake.nix`
3. Run `nix develop --command just bootstrap && nix develop --command just check`

## Project Structure

```
src/
└── main.rs
```

## Development

Run `just` to see available recipes.

## Tech Stack

- [dprint](https://dprint.dev/)
- [Nix](https://nixos.org/)
- [direnv](https://direnv.net/)
- [just](https://github.com/casey/just)

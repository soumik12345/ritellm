# format
cargo fmt --all
# lint + auto-fix what’s safe
cargo clippy --fix --allow-dirty --allow-staged
# (optional) apply compiler suggestions
cargo fix --allow-dirty --allow-staged

# format python code
ruff format
# lint + auto-fix what’s safe
ruff check --fix

export RUST_BACKTRACE := "1"

default:
    @just --list

test:
    cargo t -r

test-mod MOD:
    cargo t -rF {{MOD}} --no-default-features -- --show-output

matmul:
    @just test-mod matmul

sort:
    @just test-mod sort
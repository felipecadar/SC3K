# On remote machine with port forwarding
uv run rerun --serve-grpc

# On local machine receiving data
uv run rerun --connect rerun+http://127.0.0.1:9876/proxy
import zmq, time
from mano_stream_protocol import unpack_mano

def main():
    ctx = zmq.Context.instance()
    sock = ctx.socket(zmq.SUB)
    sock.setsockopt(zmq.SUBSCRIBE, b"mano")
    sock.setsockopt(zmq.RCVHWM, 10)     # 1도 가능, 디버그땐 10 추천
    sock.connect("tcp://127.0.0.1:5555")

    poller = zmq.Poller()
    poller.register(sock, zmq.POLLIN)

    print("[receiver] connected, waiting...", flush=True)

    while True:
        events = dict(poller.poll(1000))
        if sock not in events:
            print(".", end="", flush=True)
            continue

        # 여기서 "최신만" 남기기 위해 큐를 전부 비움 (drain)
        last_topic, last_buf = None, None
        while True:
            try:
                last_topic, last_buf = sock.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

        if last_buf is None:
            continue

        try:
            payload = unpack_mano(last_buf)
            print(
                "\n[ok]",
                "frame_id=", payload.get("frame_id"),
                "conf=", payload.get("conf"),
                "go.shape=", payload["global_orient"].shape,
                "hp.shape=", payload["hand_pose"].shape,
                flush=True
            )
        except Exception as e:
            print("\n[unpack error]", repr(e), flush=True)

if __name__ == "__main__":
    main()
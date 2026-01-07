import matplotlib.pyplot as plt

TYPE_NORMAL = 0
TYPE_CURVE = 1
TYPE_CHECKPOINT = 2
TYPE_FINISH = 3

def load_blocks(path):
    blocks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split()
            if len(parts) < 5:
                continue

            ax, ay, bx, by = map(lambda v: float(v.replace(",", ".")), parts[:4])
            t = int(parts[4])

            blocks.append((ax, ay, bx, by, t))

    return blocks


def plot_route(blocks, show=True, save_path=None):
    plt.figure()

    xs_A = [b[0] for b in blocks]
    ys_A = [b[1] for b in blocks]
    plt.plot(xs_A, ys_A)

    xs_B = [b[2] for b in blocks]
    ys_B = [b[3] for b in blocks]
    plt.plot(xs_B, ys_B)

    for (ax, ay, bx, by, t) in blocks:
        cx = (ax + bx) / 2.0
        cy = (ay + by) / 2.0

        if t == TYPE_CHECKPOINT:
            plt.scatter(cx, cy, marker="s", s=50, label="Checkpoint")
        elif t == TYPE_FINISH:
            plt.scatter(cx, cy, marker="x", s=80, label="Meta")

    plt.gca().set_aspect("equal", adjustable="box")
    plt.grid(True)
    plt.legend()

    if save_path:
        plt.savefig(save_path, dpi=300)

    if show:
        plt.show()


if __name__ == "__main__":
    input_txt = "..\\..\\maps\\Test2.Map.txt"
    blocks = load_blocks(input_txt)
    plot_route(blocks)

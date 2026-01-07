from pygbx import Gbx, GbxType
import csv
import sys
from pathlib import Path


def export_map_info(path: str = "C:\\Users\\piotr\\Documents\\Trackmania\\Maps\\My Maps\\Test2.Map.Gbx"):
    path = Path(path)
    if not path.exists():
        print(f"Plik {path} nie istnieje.")
        return

    g = Gbx(str(path))

    # Szukamy klasy mapy: CHALLENGE (TMNF/TMUF/TM2 itp.) :contentReference[oaicite:1]{index=1}
    challenge = g.get_class_by_id(GbxType.CHALLENGE)
    if not challenge:
        quit()

    print(f'Map Name: {challenge.map_name}')
    print(f'Map Author: {challenge.map_author}')
    print(f'Environment: {challenge.environment}')



if __name__ == "__main__":
    export_map_info()

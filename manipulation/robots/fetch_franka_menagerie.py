from __future__ import annotations

"""Fetch the official MuJoCo Menagerie Panda and build a slim-tool variant.

The seven-link Panda arm, inertias, joint limits, collision meshes and seven arm
position servos are kept from the upstream model.  Only the stock hand/finger
subtree is replaced locally by a small rigid bridge and one short central finger.
"""

from pathlib import Path
import argparse
import urllib.request
import xml.etree.ElementTree as ET

MENAGERIE_REF = "v2026.9.2"
BASE_URL = (
    "https://raw.githubusercontent.com/google-deepmind/"
    f"mujoco_menagerie/{MENAGERIE_REF}/franka_emika_panda"
)
DEFAULT_DIR = Path(__file__).resolve().parent / "assets" / "franka_emika_panda"


def _download(url: str, dst: Path, *, force: bool = False) -> None:
    if dst.exists() and not force:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "manipulation-franka-mppi/1"})
        with urllib.request.urlopen(request, timeout=60) as response, tmp.open("wb") as f:
            while True:
                chunk = response.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
        tmp.replace(dst)
    except Exception:
        tmp.unlink(missing_ok=True)
        raise


def _find_body(root: ET.Element, name: str) -> ET.Element | None:
    for body in root.findall(".//body"):
        if body.attrib.get("name") == name:
            return body
    return None


def _build_slim_variant(panda_xml: Path, output: Path) -> Path:
    tree = ET.parse(panda_xml)
    root = tree.getroot()
    root.set("model", "panda_single_finger")

    link7 = _find_body(root, "link7")
    if link7 is None:
        raise RuntimeError("upstream Panda XML is missing link7")
    for child in list(link7):
        if child.tag == "body" and child.attrib.get("name") == "hand":
            link7.remove(child)

    tool = ET.SubElement(
        link7,
        "body",
        {
            "name": "slim_gripper",
            "pos": "0 0 0.107",
            "quat": "0.9238795 0 0 -0.3826834",
        },
    )
    ET.SubElement(
        tool,
        "inertial",
        {"mass": "0.18", "pos": "0 0 0.075", "diaginertia": "0.00045 0.00045 0.00012"},
    )
    # Very small bridge: no bulky palm/head.
    ET.SubElement(
        tool,
        "geom",
        {
            "name": "slim_bridge",
            "type": "box",
            "pos": "0 0 0.014",
            "size": "0.012 0.030 0.014",
            "rgba": "0.82 0.84 0.86 1",
            "friction": "1.2 0.01 0.001",
        },
    )
    # One short central finger. Its local +Z direction is the tool axis.
    # The distal reach is roughly half the previous two-finger tool.
    ET.SubElement(
        tool,
        "geom",
        {
            "name": "slim_finger",
            "type": "capsule",
            "fromto": "0 0 0.020 0 0 0.080",
            "size": "0.006",
            "rgba": "0.90 0.91 0.92 1",
            "friction": "1.2 0.01 0.001",
        },
    )
    ET.SubElement(
        tool,
        "geom",
        {
            "name": "slim_tip",
            "type": "sphere",
            "pos": "0 0 0.086",
            "size": "0.006",
            "rgba": "0.20 0.20 0.22 1",
            "friction": "1.4 0.01 0.001",
        },
    )
    ET.SubElement(
        tool,
        "site",
        {"name": "slim_tip_mid", "pos": "0 0 0.086", "size": "0.004", "rgba": "0.1 1 0.1 0.35"},
    )

    # The stock hand joints/tendon/actuator are gone.  Keep only the seven arm
    # servos and a seven-joint home keyframe.
    for section_name in ("tendon", "equality"):
        section = root.find(section_name)
        if section is not None:
            root.remove(section)

    actuator = root.find("actuator")
    if actuator is None:
        raise RuntimeError("upstream Panda XML is missing actuator section")
    for child in list(actuator):
        if child.attrib.get("name") == "actuator8" or "tendon" in child.attrib:
            actuator.remove(child)

    keyframe = root.find("keyframe")
    if keyframe is not None:
        for key in keyframe.findall("key"):
            if key.attrib.get("name") == "home":
                key.set("qpos", "0 0 0 -1.57079 0 1.57079 -0.7853")
                key.set("ctrl", "0 0 0 -1.57079 0 1.57079 -0.7853")

    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    output.parent.mkdir(parents=True, exist_ok=True)
    tree.write(output, encoding="utf-8", xml_declaration=False)
    return output


def fetch_franka_menagerie(destination: str | Path = DEFAULT_DIR, *, force: bool = False) -> Path:
    root = Path(destination)
    root.mkdir(parents=True, exist_ok=True)

    panda_xml = root / "panda.xml"
    _download(f"{BASE_URL}/panda.xml", panda_xml, force=force)
    for name in ("README.md", "LICENSE"):
        _download(f"{BASE_URL}/{name}", root / name, force=force)

    xml_root = ET.parse(panda_xml).getroot()
    mesh_files = sorted(
        {
            mesh.attrib["file"]
            for mesh in xml_root.findall(".//mesh")
            if mesh.attrib.get("file")
        }
    )
    for filename in mesh_files:
        _download(
            f"{BASE_URL}/assets/{filename}",
            root / "assets" / filename,
            force=force,
        )

    return _build_slim_variant(panda_xml, root / "panda_slim.xml")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch the upstream Panda and generate the slim single-finger variant."
    )
    parser.add_argument("--destination", type=Path, default=DEFAULT_DIR)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    path = fetch_franka_menagerie(args.destination, force=args.force)
    print(path)


if __name__ == "__main__":
    main()

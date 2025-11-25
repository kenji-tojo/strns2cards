import bpy
import os
import sys
import argparse

# -------- CLI args --------
def parse_args():
    argv = sys.argv[sys.argv.index("--") + 1:] if "--" in sys.argv else []
    p = argparse.ArgumentParser(add_help=False)
    p.add_argument("--dataset", default="CT2Hair")
    p.add_argument("--experiment_ID", default="example")
    p.add_argument("--config", default="cluster300")
    return p.parse_args(argv)

args = parse_args()

# === Settings
dataset = args.dataset
experiment_ID = args.experiment_ID
config = args.config
output_root = os.path.abspath(os.path.join(f"./output/{experiment_ID}", dataset, config))
# output_root = os.path.abspath(os.path.join(f"./output/avatar/Revision/Resample1"))
model_names = sorted([d for d in os.listdir(output_root) if os.path.isdir(os.path.join(output_root, d))])

# === Helpers
def clean_scene():
    bpy.ops.wm.read_factory_settings(use_empty=True)

def import_obj(obj_path):
    bpy.ops.wm.obj_import(filepath=obj_path)

def set_smooth_shading(obj):
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.shade_smooth()

def export_fbx(fbx_path):
    bpy.ops.export_scene.fbx(
        filepath=fbx_path,
        use_selection=True,
        apply_unit_scale=True,
        object_types={'MESH'},
        bake_space_transform=False,
        mesh_smooth_type='FACE',
    )

# === Batch conversion ===
for model_name in model_names:
    obj_path = os.path.join(output_root, model_name, "strip.obj")
    if not os.path.exists(obj_path):
        continue
    out_fbx_path = os.path.join(output_root, model_name, "strip.fbx")

    print(f"🔄 Converting {os.path.basename(obj_path)} → {os.path.basename(out_fbx_path)}")

    clean_scene()
    import_obj(obj_path)

    objs = bpy.context.selected_objects
    if not objs:
        print(f"❌ Failed to import: {obj_path}")
        continue

    obj = objs[0]
    set_smooth_shading(obj)
    export_fbx(out_fbx_path)

    print(f"✅ Saved: {out_fbx_path}")

print("🎉 All CT2Hair aligned head models converted to FBX.")

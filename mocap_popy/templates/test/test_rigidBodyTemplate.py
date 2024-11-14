# %%

rbt = RigidBodyTemplate(
    name="foot",
    parent_models=["subject"],
    markers=["heel", "toe"],
    segments=["heel-toe"],
    param_index_marker_mapping={"1": "heel", "2": "toe"},
    tolerances={
        "segment": {"heel-toe": 0.1},
        "joint": {"heel": 0.1, "toe": 0.1},
    },
)

rbt2 = RigidBodyTemplate(
    name="foot",
    parent_models=["subject"],
    markers=["heel", "toe"],
    segments=["heel-toe"],
)

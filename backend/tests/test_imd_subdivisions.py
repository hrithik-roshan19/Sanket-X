from app.data_sources.imd_subdivisions import IMD_SUBDIVISION_NAMES, SUBDIVISIONS, slugify


def test_imd_registry_has_exactly_36_unique_names():
    assert len(IMD_SUBDIVISION_NAMES) == 36
    assert len({x.casefold() for x in IMD_SUBDIVISION_NAMES}) == 36
    assert len(SUBDIVISIONS) == 36


def test_imd_registry_ids_and_slugs_are_unique():
    assert len({x.subdivision_id for x in SUBDIVISIONS}) == 36
    assert len({x.slug for x in SUBDIVISIONS}) == 36
    assert slugify("Assam & Meghalaya") == "assam_meghalaya"

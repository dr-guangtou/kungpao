from __future__ import absolute_import, division, print_function

from kungpao.display import science_cmap

def test_science_cmap():
    """Test the sceince_cmap function."""
    cmap_list = ['bamako', 'batlow', 'berlin', 'bilbao',
                 'davos', 'imola', 'lajolla', 'lapaz',
                 'oslo', 'roma', 'vik']
    
    cmap_test = science_cmap(cmap_name='davos')
    assert cmap_test.name == 'davos'

    assert set(cmap_list) == set(science_cmap(list_maps=True))
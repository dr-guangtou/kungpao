# KungPao (宫保) - Delicious Galaxies!

<img src="doc/kungpao_logo.png" width="60%">

------

Besides being one of the [signature Chinese dishes](https://en.wikipedia.org/wiki/Kung_Pao_chicken), `kungpao` can also help you deal with photometry of galaxies from recent imaging surveys (e.g. HSC, DECaLS).

Also, `kungpao` does not stand for anything because forced acronym is for psychopath.

Recent Updates
--------------

Applications
------------

- Multi-stage objects detection and deblending
- Measure and subtract 2-D sky background mode
- Generate object masks for photometry
- Model 2-D light distribution of objects on the image

Installation
------------

- `python setup.py install` or `python setup.py develop` will do the job.
- Right now, `kungpao` only supports `Python>=3`.  If you are still using `Python 2`, you should make the switch.
- `kungpao` only depends on `numpy`, `scipy`, `astropy`, `sep`, `astroquery`, and `matplotlib`. All can be installed using `pip` or `conda`.

Documents
---------

I <del>promise</del>hope that documents will be available soon...but right now, please take a look at the Jupyter Notebook [demos](https://github.com/dr-guangtou/kungpao/tree/master/demo) for each functionality.


Acknowledgement
---------------


Reporting bugs
--------------

If you notice a bug in `kungpao` (and you will~), please file an detailed issue at:

https://github.com/dr-guangtou/kungpao/issues



Requesting features
-------------------

If you would like to request a new feature, do the same thing.


License
-------

Copyright 2019 Song Huang and contributors.

`kungpao` is free software made available under the MIT License. For details see
the LICENSE file.

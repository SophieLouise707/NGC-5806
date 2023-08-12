# NGC-5806
Python code, mostly involving PPXF, used to analyze MUSE data of the galaxy NGC 5806

Files in this repository include:
-Binning for 5806.ipynb - Jupyter notebook applying Voronoi binning to a MUSE IFU data cube, to output a re-binned fits file
-Spyder_PPXF_multispec.py - Python code, which I run on Spyder, to apply PPXF to each bin in a Voronoi-binned IFU data cube. Does a full fit for stellar continuum and gas emission lines. Saves results in a fits file, and a fits file with the bestfit for each bin. Uses XShooter release 2 stellar templates. 
-Spyder_miles_PPXF_multispec.py - Code for Spyder which also applies PPXF to each bin of a Voronoi binned data cube, but only fits the stellar continuum. Uses miles Vazdekis stellar templates. Saves weights for each stellar template to a fits file--used for mapping stellar population and metallicity
-Stellar_pop_PPXF.ipynb - Jupyter notebook to map results from Spyder_miles_PPXF_multispec.py.

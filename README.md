This repository was created by Daniela Täuber in 2020.

Contributions so far have been made by Sebastian Unger, René Lachmann, Rainer Heintzmann, Meng Luo, and Mohammad Soltaninezhad

It aims at developing tools for processing spectral data from vibrational nanospectroscopy methods, 
and conventional FTIR (and Raman spectroscopy) for comparison.

So far experimental methods have been:
- photo-induced force microscopy (PiFM)
- FTIR
- Raman spectroscopy
- optical photothermal infrared spectroscopy (O-PTIR)


**Subtopics are sorted in folders:**
1. NanIRim: processing and comparing scan images (mainly PiFM)

2. NanIRspec: processing and analyzing spectral data:  
-  _hyPIRana_: analysis of PiFM hyperspectral data (providing callibration, mean spectra, PCA); Applicable to one complete spetral region (1 laser tuner)
-  _AreaSelect_ allows to choose a spacial region of interest via a gui
-  _hyPIRana_SplitRange_: Other than _hyPIRana_ it allows to split the data set into two spectral ranges which will be analyzed independently (two laser tuners)
-  _monIRana_: analysis of already callibrated single spectra (providing mean spectra, PCA)

3. SpecTools: basic tools for background subtraction etc.


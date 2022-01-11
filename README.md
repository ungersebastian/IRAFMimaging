This repository was created by Daniela Täuber in 2020.

Contributions so far have been made by Sebastian Unger, René Lachmann, Rainer Heintzmann, Meng Luo, and Mohammad Soltaninezhad

It aims at developing tools for processing spectral data from vibrational nanospectroscopy methods, 
and conventional FTIR (and Raman spectroscopy) for comparison.

So far experimental methods have been:
- photo-induced force microscopy (PiFM)
- FTIR
- Raman spectroscopy
- optical photothermal infrared spectroscopy (O-PTIR)

Goals for publication:
1. TDA on Bacillus subtilis
- The scientific questions is to show: PiFM can reveal the local interaction of Vancomycin with the bacteria wall & TDA is a suitable tool to reveal this in the hyperspectral data
- The interaction of the antibiotics Vancomycin with the bateria wall of Bacillus subtilis is well-studied. The spectral areas of interaction are seen in the difference spectra described in Robins Master Thesis which can be found in „Reports“
- We have also spectra of Vancomycin and of peptidoglycan, as well as of D-Ala-D-Ala main ingredients of the interaction.
- One advantage of TDA in this case is the sensitivity to small changes in small data sets

2. TDA on human RPE granules (Retina)
- The scientific question here is: PIFM can show reasonable differences between single granules. The method is suitable to study surface variations of granules related to age dependent degeneration
- The surface chemistry of the granules is not fully known.
- We have single spectra of some relevant chemicals, in particular one type of melanin and of retinal (see Robin’s internship report) 

**Subtopics are sorted in folders:**
1. NanIRim: processing and comparing scan images (mainly PiFM)

2. NanIRspec: processing and analyzing spectral data:  
-  _hyPIRana_: analysis of PiFM hyperspectral data (providing callibration, mean spectra, PCA); Applicable to one complete spetral region (1 laser tuner)
-  _AreaSelect_ allows to choose a spatial region of interest via a gui
-  _hyPIRana_SplitRange_: Other than _hyPIRana_ it allows to split the data set into two spectral ranges which will be analyzed independently (two laser tuners)
-  _monIRana_: analysis of already callibrated single spectra (providing mean spectra, PCA)

3. SpecTools: basic tools for background subtraction etc.

**Ressources in folder ressources in NanIRspec:**

Retina (many laser failures in those spectra):
- 3 data sets on Ret 24 obtained on succeeding days. Two options for CaF2 substrate correction possible, but only the so far better one of the substrate calibration files uploaded
- 2 data sets on ret29 from a very dry sample obtained on succeeding days. Two different QCL chips used for those hyperspectral scans. They failed not at the same time. So a suggestion would be to just analyse the matching range to Ret24 and leave the other one out

Bacillus subtilis (data sets have different spectral range:
- 2108_BacVan30_1400-1659cm-1 — CaF2 file for calibration available — best data set! —
- 2108_Control30_1400-1659cm-1 — CaF2 file for calibration available
- 2107_BacVan30-weak_989-1349cm-1 — CaF2 file for calibration available — vey weak data set
- 2105_BacVan15_1490-1659cm-1 - CaF2 file for calibration not yet ready
- 2105_BacVan30_989-1349cm-1 — CaF2 file for calibration not yet ready
- 2105_BacVan30_1351-1659cm-1 — CaF2 file for calibration not yet ready


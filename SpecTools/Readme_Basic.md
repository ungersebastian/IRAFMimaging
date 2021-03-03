This folder (will) contain basic tools:

**baselinepolyfit.py**
_Authored by Mohammad Soltaninezhad_
This program is designed to correct baseline with polynomial method and subtracting old spectrum with fitted
polynomial graph.
1. plot raw spectrum so client will be able to decide the best method (have a fit on whole spectrum or choose a
desire wavelength)
2. Program asks if you want whole spec fitting or get desired regions
3. If multiregional fitting is selected, the program will ask how many sections are needed
4. Every region can be fitted with desired polynomial degree (#line 68 and 69 of code ???)
5. Separated regions recombine in one plot in the end
6. Data export?


- normalisation

- smoothing spectra

- deconvolution using known components: [two Components](https://stackoverflow.com/questions/63003805/determining-relative-contribution-of-two-components-to-a-measured-spectrum)

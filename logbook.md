#Log Book for project:

##Links:
###Papers
- 1. Levy State Space Model: https://arxiv.org/pdf/1912.12524.pdf
- 2. LSSM for tracking: http://publications.eng.cam.ac.uk/1303074/
- 3. HF futures returns using Langevin: http://publications.eng.cam.ac.uk/455734/
- 4. inverse Gaussian processes: https://arxiv.org/pdf/2105.09429.pdf

###Data sources
- 1. Possible gas data source: https://tradingeconomics.com/commodity/natural-gas

##14/07/21
- Implemented a gamma process according to paper 1
TO DO
- DONE: ts process
- DONE: reduce repeated code and make more modular
- speed up for loops

##18/10/21
- Implemented the ts process according to paper 1
- Tidied up the code into a more modular form
TO DO
- DONE: check stability of the xi's for both processes, currently liable to overflow
- DONE: get access to the papers
- move on to the GIG process

##Meeting 19/10/21
- Normal Gamma process
- beta = gamma^2/2
- need to look for some data - investigate CME gas futures

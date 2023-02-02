# k-anonymization
This project aimed to implement k-anonymization algorithms in Python and compare their performance by executing your algorithms on a real dataset.
A sample dataset and Domain Generalization Hierarchy (DGH) folder are provided. 

Each file in the DGHs folder contains the domain generalization hierarchy of one of the Quasi-identifier (QI) attributes, e.g., age.txt contains the DGH for the age attribute, education.txt contains the DGH for the education attribute, and so forth. The tabs indicate parent-child relationships in a DGH. For example, here is how workclass.txt represents the visual DGH of the workclass attribute:
<img width="817" alt="Screen Shot 2023-02-02 at 11 13 31" src="https://user-images.githubusercontent.com/66200657/216268299-15ea78e2-07d0-46a1-a484-527ec738a6b3.png">


## Comparison
You can find MD Cost, LM Cost and Time Cost of algorithms with different k values below. 
<img width="807" alt="Screen Shot 2023-02-02 at 10 48 11" src="https://user-images.githubusercontent.com/66200657/216263901-b50d2b6a-7c11-4445-abb2-87c92884375a.png">
<img width="805" alt="Screen Shot 2023-02-02 at 10 53 07" src="https://user-images.githubusercontent.com/66200657/216264712-b9af34da-8a1d-4762-a0fd-7a220b825f66.png">
<img width="808" alt="Screen Shot 2023-02-02 at 10 56 59" src="https://user-images.githubusercontent.com/66200657/216264910-4d1a428e-0529-490e-a5de-c02a88fff00a.png">

<img width="643" alt="Screen Shot 2023-02-02 at 10 59 55" src="https://user-images.githubusercontent.com/66200657/216265465-b20f018b-b0c1-4bd3-a504-7e9accc9dedb.png">
<img width="641" alt="Screen Shot 2023-02-02 at 11 00 37" src="https://user-images.githubusercontent.com/66200657/216265608-6991b10b-9481-4ef6-9826-458ca6e23e56.png">
<img width="634" alt="Screen Shot 2023-02-02 at 11 01 15" src="https://user-images.githubusercontent.com/66200657/216265716-cdd0c80a-5d9b-42c7-a2ab-b6a2dce57bb6.png">

## Discussion
There is a trade-off between privacy and utility. Randomized anonymizer is the fastest. The anonymizer which has the lowest utility loss is clustering-based low k, bottom-up for high k values. I would prefer to use clustering for low k values because it has the lowest MD and LM costs. Also, bottom-up can be used for high k values if high time cost is not a problem. I do not prefer to use randomized because it has very low utility loss since it puts records to ECs randomly. Results almost fit my expectations. I learned to calculate MD and LM costs, implement data anonymization algorithms, and compare performances of the algorithms from this assignment. 

## Run k-anonymization locally

### Step 1: clone the project
    git clone https://github.com/iremkrc/k-anonymization.git
    cd k-anonymization
    
### Step 2: install needed packages if they are not exist
    pip install -r requirements.txt
    
### Step 3: run the project
    python3 main.py algorithm DGH-folder raw-dataset.csv anonymized.csv k seed(for random only)
Where algorithm is one of [clustering, random, bottomup]
  

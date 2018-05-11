# plagiarism_detection_pan2015
Plagiarism Detection Approach for PAN 2015 Text Alignment task
This system is the implementation as detailed in [1] and [2] for the Text Alignment task at PAN 2015

1. REQUIREMENTS
---------------------------------------------------------
To use the algorithm you need to install the following python modules:
- PyStemmer 1.3.0 -> https://pypi.python.org/pypi/PyStemmer
- nltk -> http://www.nltk.org/


2. USAGE
---------------------------------------------------------
```python
python PAN2015 <pairs> <source document folder> <suspicious document folder> <output folder>
```

Example:
```python
python PAN2015_JCR.py E:/text-alignment/pan13-text-alignment-training-dataset-2013-01-21/pairs E:/text-alignment/pan13-text-alignment-training-dataset-2013-01-21/src E:/text-alignment/pan13-text-alignment-training-dataset-2013-01-21/susp C:/Users/sanchezperez15/Results
```

3. INPUT
---------------------------------------------------------
- <pair> It is a file containing the pairs of documents to be compare
- <source document folder> Folder with all the source documents mentioned in <pairs>
- <suspicious document folder> Folder with all the suspicios documents mentioned in <pairs>
- <output folder> Folder were the resulting xml files will be store


4. OUTPUT
---------------------------------------------------------
The results are store in the <output folder> as a xml file with the following format as required at PAN 2015 [3]:

```xml
<document reference="suspicious-documentXYZ.txt">
<feature
  name="detected-plagiarism"
  this_offset="5"
  this_length="1000"
  source_reference="source-documentABC.txt"
  source_offset="100"
  source_length="1000"
/>
<feature ... />
...
</document>
```

For example, the above file would specify an aligned passage of text between suspicious-documentXYZ.txt and source-documentABC.txt, and that it is of length 1000 characters, starting at character offset 5 in the suspicious document and at character offset 100 in the source document.


5. NOTE
---------------------------------------------------------
In the main method the following lines allow comparing 2 documents:

```python
sgsplag_obj = SGSPLAG(read_document(<path_to_suspicious_document>), read_document(<path_to_source_document>), parameters)
type_plag, summary_flag = sgsplag_obj.process()
```

where the results are stored in <sgsplag_obj.detections>.

We state this note in order to facilitate the reusing of this method outside the PAN requirements


6. REFERENCES
---------------------------------------------------------
[1] Sanchez-Perez, M.A., Gelbukh, A., Sidorov, G.: Adaptive algorithm for plagiarism detection: The best-performing approach at PAN 2014 text alignment competition. In: Mothe, J., Savoy, J., Kamps, J., Pinel-Sauvagnat, K., Jones, G.J.F., SanJuan, E., Cappellato, L., Ferro, N. (eds.) Experimental IR Meets Multilinguality, Multi-modality, and Interaction - 6th International Conference of the CLEF Association, CLEF 2015, Toulouse, France, September 8-11, 2015, Proceedings. Lecture Notes in Computer Science, vol. 9283, pp. 402{413. Springer (2015)

[2] Sanchez-Perez, M.A., Gelbukh, A.F., Sidorov, G.: Dynamically adjustable approach through obfuscation type recognition. In: Cappellato, L., Ferro, N., Jones, G.J.F., SanJuan, E. (eds.) Working Notes of CLEF 2015 - Conference and Labs of the Evaluation forum, Toulouse, France, September 8-11, 2015. CEUR Workshop Proceedings, vol. 1391. CEUR-WS.org (2015), http://ceur-ws.org/Vol-1391/92-CR.pdf

[3] http://pan.webis.de/clef15/pan15-web/plagiarism-detection.html


*** For more questions do not hesitate to contact us! ***

- Miguel Ángel Sánchez Pérez <miguel.sanchez.nan(?)gmail.com>
- Alexander Gelbukh <gelbukh(?)gelbukh.com>
- Grigori Sidorov <sidorov(?)cic.ipn.mx>


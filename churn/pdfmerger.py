import os
from PyPDF2 import PdfFileMerger

path = 'C:/Users/Madhivarman/Desktop/NDA'
pdf_files = os.listdir(path)

merger = PdfFileMerger()

for pdf in pdf_files:
	full_path = path + "/" + pdf
	merger.append(open(full_path, 'rb'))

with open('result.pdf', 'wb') as fout:
    merger.write(fout)
# -*- coding: utf-8 -*-
"""
File Name: Compile_Diary.py
Purpose: Combine diary entries into a single pdf
Author: Samuel Wong
"""
from PyPDF2 import PdfFileMerger
import datetime

def get_date_list(end_date):
    pdfs = [] #initialize a list of pdf title
    start = datetime.datetime.strptime("19-12-2019", "%d-%m-%Y")
    end = datetime.datetime.strptime(end_date, "%d-%m-%Y")
    date_generated = [start + datetime.timedelta(days=x) for x in range(0,
                      (end-start).days)]
    for date in date_generated:
        file_title = date.strftime("%b_%d__%Y.pdf").replace("_0", "_")
        pdfs.append(file_title)
    return pdfs

merger = PdfFileMerger()
pdfs = get_date_list("27-01-2020")

for pdf in pdfs:
    merger.append(pdf)
    
merger.write("../Diary.pdf")
merger.close()

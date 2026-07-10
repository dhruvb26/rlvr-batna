TEX = pdflatex -interaction=nonstopmode
BIB = bibtex
SRC = paper

.PHONY: all rebuild clean

all: $(SRC).pdf

$(SRC).pdf: $(SRC).tex references.bib
	$(TEX) $(SRC)
	$(BIB) $(SRC)
	$(TEX) $(SRC)
	$(TEX) $(SRC)

rebuild:
	$(TEX) $(SRC)
	$(BIB) $(SRC)
	$(TEX) $(SRC)
	$(TEX) $(SRC)

clean:
	rm -f $(SRC).{aux,bbl,blg,log,out,toc,lof,lot,fls,fdb_latexmk}

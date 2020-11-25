TESTDIR:=./output/test/
TABLEDIR:=./output/tex/
PLOTDIR:=./output/plot/
REPORT:=./report/report.pdf

# find the tables and plots that can be made using the results
TABLES=$(wildcard $(TABLEDIR)*/table.tex)
SCATTER=$(wildcard $(PLOTDIR)*/scatter.pdf)
PARETO=$(wildcard $(PLOTDIR)*/pareto.tex)

all: $(REPORT)

%.pdf: %.tex $(TABLES) $(SCATTER) $(PARETO)
	export TEXINPUTS="./report/theme:$$TEXINPUTS"
	pdflatex -shell-escape -output-directory ./report $<

$(TABLEDIR)%/table.tex: plot.py $(wildcard $(TESTDIR)%/*/test.csv) plot

$(PLOTDIR)%/scatter.pdf: plot.py $(wildcard $(TESTDIR)%/*/test.csv) plot

$(PLOTDIR)%/pareto.pdf: plot.py $(wildcard $(TESTDIR)%/*/test.csv) plot

.PHONY: plot deptest
plot:
	./plot.py

FORCE:
	


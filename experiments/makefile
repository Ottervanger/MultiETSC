TESTDIR:=./output/test/
TABLEDIR:=./output/tex/
PLOTDIR:=./output/plot/
REPORT:=./report/report.pdf

# find the tables and plots that can be made using the results
TABLES=$(patsubst %,$(TABLEDIR)%/table.tex, $(shell ls -1 ./output/test/ ))
PLOTS=$(patsubst %,$(PLOTDIR)%/scatter.pdf, $(shell ls -1 ./output/test/ ))
BASELINES=$(patsubst %,./output/baseline/%.csv, $(shell ls -1 ./output/test/ ))

all: $(REPORT)

%.pdf: %.tex $(TABLES) $(PLOTS) FORCE
	export TEXINPUTS="./report/theme:$$TEXINPUTS"
	pdflatex -shell-escape -output-directory ./report $<

$(TABLEDIR)%/table.tex: plot.py $(BASELINES) $(wildcard $(TESTDIR)%/*/test.csv)
	./plot.py

$(PLOTDIR)%/scatter.pdf: plot.py $(BASELINES) $(wildcard $(TESTDIR)%/*/test.csv)
	./plot.py

$(PLOTDIR)%/pareto.pdf: plot.py $(BASELINES) $(wildcard $(TESTDIR)%/*/test.csv)
	./plot.py

./output/baseline/%.csv:
	./baseline.py

FORCE:
	

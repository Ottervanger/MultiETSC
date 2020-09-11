TESTDIR:=./output/test/
TABLEDIR:=./output/tex/
PLOTDIR:=./output/plot/
REPORT:=./report/report.pdf
# SLIDES:=./report/slides.pdf

# find the tables and plots that can be made using the results
TABLES=$(wildcard $(patsubst %,$(TABLEDIR)%/table.tex,$(shell ls -1 ./output/test/ )))
PLOTS=$(wildcard $(patsubst %,$(PLOTDIR)%/boxplot.pdf,$(shell ls -1 ./output/test/ )))

all: $(REPORT) $(SLIDES)

%.pdf: %.tex $(TABLES) $(PLOTS)
	export TEXINPUTS="./report/theme:$$TEXINPUTS"
	pdflatex -shell-escape -output-directory ./report $<

.SECONDEXPANSION:

$(TABLEDIR)%.tex: plot.py $$(wildcard $(TESTDIR)$$(notdir $$(basename $$@))/*/test.csv)
	./plot.py

$(PLOTDIR)%.pdf: plot.py $$(wildcard $(TESTDIR)$$(notdir $$(basename $$@))/*/test.csv)
	./plot.py


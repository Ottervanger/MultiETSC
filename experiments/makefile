RESULTDIR:=./output/results/
TABLEDIR:=./output/tex/
PLOTDIR:=./output/plot/
REPORT:=./report/report.pdf
SLIDES:=./report/slides.pdf

# find the tables and plots that can be made using the results
TABLES=$(wildcard $(patsubst %,$(TABLEDIR)%,$(shell ls -1 ./output/results/ | perl -pe "s/(.*?)-.*-(.*?)\..*/\1-\2.\*/" | sort | uniq)))
PLOTS=$(wildcard $(patsubst %,$(PLOTDIR)%,$(shell ls -1 ./output/results/ | perl -pe "s/(.*?)-.*-(.*?)\..*/\1-\2.\*/" | sort | uniq)))

all: $(REPORT) $(SLIDES)

%.pdf: %.tex $(TABLES) $(PLOTS)
	@export TEXINPUTS=.:./report/theme:$$TEXINPUTS
	pdflatex -shell-escape -output-directory ./report $< > /dev/null

.SECONDEXPANSION:

$(TABLEDIR)%.tex: plot.py $$(wildcard $(RESULTDIR)$$(subst -,*,$$(notdir $$(basename $$@)))*)
	./plot.py $(subst -,*,$(notdir $(basename $@)))*

$(PLOTDIR)%.pdf: plot.py $$(wildcard $(RESULTDIR)$$(subst -,*,$$(notdir $$(basename $$@)))*)
	./plot.py $(subst -,*,$(notdir $(basename $@)))*


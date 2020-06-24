RESULTDIR:=./output/results/
TABLEDIR:=./output/tex/
PLOTDIR:=./output/plot/
REPORT:=./output/report/report.pdf
TABLES=$(wildcard $(patsubst %,$(TABLEDIR)%,$(shell ls -1 output/results/ | perl -pe "s/(.*?)-.*-(.*?)\..*/\1-\2.\*/" | sort | uniq)))
PLOTS=$(wildcard $(patsubst %,$(PLOTDIR)%,$(shell ls -1 output/results/ | perl -pe "s/(.*?)-.*-(.*?)\..*/\1-\2.\*/" | sort | uniq)))


.SECONDEXPANSION:
	
all: $(REPORT)

$(REPORT): report.tex $(TABLES) $(PLOTS)
	pdflatex -shell-escape -output-directory output/report/ report.tex > /dev/null

$(TABLEDIR)%.tex: $$(wildcard $(RESULTDIR)$$(subst -,*,$$(notdir $$(basename $$@)))*)
	./plot.py $(subst -,*,$(notdir $(basename $@)))*

$(PLOTDIR)%.pdf: $$(wildcard $(RESULTDIR)$$(subst -,*,$$(notdir $$(basename $$@)))*)
	./plot.py $(subst -,*,$(notdir $(basename $@)))*


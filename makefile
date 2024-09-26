CORE-DIR= $(HOME)/REPOS/COSYINF-CORE
CORE = $(addsuffix .bin, cosy utilities elements)

COMMON = $(addsuffix .bin, header)
SETUPS = $(addsuffix .bin, 8PER/8TILTED 16PER/16TILTED)

define RM
	find $(1) -type f -name $(2) -print -delete
endef

core.bld: $(addprefix $(CORE-DIR)/bin/, $(CORE))
	echo $(CORE) >> core.bld
common.bld: bin/elements.bin $(addprefix bin/common/, $(COMMON))
	echo $(COMMON) >> common.bld
setups.bld: $(addprefix bin/setups/, $(SETUPS))
	echo $(SETUPS) >> setups.bld

$(CORE-DIR)/bin/cosy.bin: $(CORE-DIR)/src/cosy.fox ;
	cosy $<;
$(CORE-DIR)/bin/utilities.bin: $(CORE-DIR)/src/utilities.fox $(CORE-DIR)/bin/cosy.bin 
	cosy $<;
$(CORE-DIR)/bin/elements.bin: $(CORE-DIR)/src/elements.fox $(CORE-DIR)/bin/utilities.bin 
	cosy $<;

bin/elements.bin: src/elements.fox $(CORE-DIR)/bin/utilities.bin # local elements.fox
	cosy $<;
bin/common/%.bin: src/common/%.fox
	cosy $<;
bin/setups/%.bin: src/setups/%.fox common.bld
	cosy $<;

TE/% : test/%.fox setups.bld
	cosy $<;

clean:
	$(call RM, '.', '*.lis')

uninstall:
	$(call RM, 'bin', '*.bin')
	rm -f *.bld
	$(call RM, '.', '*.lis')

purge:
	$(call RM, 'data', '*.dat')
	$(call RM, 'data', '*.in')
	$(call RM, 'img', '*.png')
	$(call RM, 'img', '*.pdf')

dump:
	rm -v img/dump/*

.PHONY: clean uninstall purge dump

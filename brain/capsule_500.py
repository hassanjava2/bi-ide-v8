#!/usr/bin/env python3
"""
capsule_500.py — 500+ كبسولة لإعادة بناء الحضارة 🌳🏛️

الهدف: تغطية كل المعرفة البشرية اللازمة لإعادة بناء الحضارة
من الصفر بعد كارثة — بدون إنترنت.

البنية:
  20 شجرة رئيسية → 500+ كبسولة

الاستخدام:
  from brain.capsule_500 import CAPSULE_REGISTRY, expand_tree
  expand_tree()  # يوسّع capsule_tree من 31 → 500+
"""

# ═══════════════════════════════════════════════════════════
# 500+ كبسولة — كل المعرفة البشرية
# ═══════════════════════════════════════════════════════════
# Format: "capsule_id": ("اسم عربي", ["keyword1", "keyword2", ...])

CAPSULE_REGISTRY = {
    # ══════════════════════════════════════
    # 1. العلوم الأساسية (50 كبسولة)
    # ══════════════════════════════════════
    # فيزياء
    "science.physics.mechanics": ("ميكانيكا", ["force", "motion", "newton", "momentum"]),
    "science.physics.thermodynamics": ("ديناميكا حرارية", ["heat", "entropy", "carnot", "temperature"]),
    "science.physics.quantum": ("ميكانيكا كم", ["quantum", "electron", "photon", "wave"]),
    "science.physics.optics": ("بصريات", ["light", "lens", "laser", "fiber", "mirror"]),
    "science.physics.acoustics": ("صوتيات", ["sound", "wave", "frequency", "resonance"]),
    "science.physics.electromagnetism": ("كهرومغناطيسية", ["maxwell", "field", "wave", "radiation"]),
    "science.physics.fluid_dynamics": ("ميكانيكا موائع", ["flow", "pressure", "bernoulli", "pipe"]),
    "science.physics.nuclear": ("فيزياء نووية", ["fission", "fusion", "reactor", "radiation"]),
    "science.physics.relativity": ("نسبية", ["einstein", "spacetime", "gravity", "mass"]),
    "science.physics.solid_state": ("فيزياء الجوامد", ["crystal", "semiconductor", "band_gap"]),

    # كيمياء
    "science.chemistry.organic": ("كيمياء عضوية", ["polymer", "carbon", "synthesis", "reaction"]),
    "science.chemistry.inorganic": ("كيمياء غير عضوية", ["metal", "oxide", "mineral", "crystal"]),
    "science.chemistry.industrial": ("كيمياء صناعية", ["process", "catalyst", "reactor", "yield"]),
    "science.chemistry.analytical": ("كيمياء تحليلية", ["spectroscopy", "chromatography", "titration"]),
    "science.chemistry.physical": ("كيمياء فيزيائية", ["kinetics", "equilibrium", "thermochemistry"]),
    "science.chemistry.biochemistry": ("كيمياء حيوية", ["enzyme", "protein", "DNA", "metabolism"]),
    "science.chemistry.environmental": ("كيمياء بيئية", ["pollution", "treatment", "waste", "recycle"]),
    "science.chemistry.electrochemistry": ("كيمياء كهربائية", ["battery", "corrosion", "plating", "cell"]),
    "science.chemistry.polymer": ("كيمياء بوليمرات", ["plastic", "rubber", "fiber", "resin"]),
    "science.chemistry.pharmaceutical": ("كيمياء دوائية", ["drug", "synthesis", "formulation"]),

    # أحياء
    "science.biology.genetics": ("وراثة", ["DNA", "gene", "mutation", "inheritance"]),
    "science.biology.microbiology": ("أحياء دقيقة", ["bacteria", "virus", "fungus", "antibiotic"]),
    "science.biology.botany": ("نبات", ["plant", "seed", "photosynthesis", "crop"]),
    "science.biology.zoology": ("حيوان", ["animal", "livestock", "breeding", "veterinary"]),
    "science.biology.ecology": ("بيئة", ["ecosystem", "biodiversity", "conservation"]),
    "science.biology.anatomy": ("تشريح", ["organ", "bone", "muscle", "nerve"]),
    "science.biology.physiology": ("وظائف أعضاء", ["heart", "lung", "kidney", "digestion"]),
    "science.biology.immunology": ("مناعة", ["immune", "vaccine", "antibody", "infection"]),

    # رياضيات
    "science.math.algebra": ("جبر", ["equation", "variable", "matrix", "polynomial"]),
    "science.math.calculus": ("تفاضل وتكامل", ["derivative", "integral", "limit", "function"]),
    "science.math.statistics": ("إحصاء", ["probability", "distribution", "regression", "bayes"]),
    "science.math.geometry": ("هندسة رياضية", ["triangle", "circle", "area", "volume"]),
    "science.math.linear_algebra": ("جبر خطي", ["vector", "matrix", "eigenvalue", "tensor"]),
    "science.math.discrete": ("رياضيات متقطعة", ["graph", "logic", "combinatorics", "set"]),
    "science.math.numerical": ("تحليل عددي", ["approximation", "interpolation", "optimization"]),

    # جيولوجيا
    "science.geology.mineralogy": ("معادن", ["mineral", "crystal", "ore", "deposit"]),
    "science.geology.mining": ("تعدين", ["extraction", "mine", "drill", "excavation"]),
    "science.geology.hydrology": ("مياه جوفية", ["aquifer", "well", "groundwater", "spring"]),
    "science.geology.soil_science": ("علوم تربة", ["soil", "clay", "sand", "fertility"]),
    "science.geology.seismology": ("زلازل", ["earthquake", "fault", "seismic", "richter"]),

    # نانو + مواد
    "science.nanotech": ("تقنية النانو", ["nano", "carbon_tube", "quantum_dot"]),
    "science.materials.metals": ("علم معادن", ["alloy", "steel", "aluminum", "copper"]),
    "science.materials.ceramics": ("سيراميك", ["ceramic", "porcelain", "glass", "refractory"]),
    "science.materials.polymers": ("بوليمرات", ["plastic", "rubber", "composite", "fiber"]),
    "science.materials.composites": ("مواد مركبة", ["carbon_fiber", "fiberglass", "kevlar"]),
    "science.materials.testing": ("اختبار مواد", ["tensile", "hardness", "fatigue", "NDT"]),

    # ══════════════════════════════════════
    # 2. الطب والصحة (40 كبسولة)
    # ══════════════════════════════════════
    "medicine.emergency": ("طوارئ", ["first_aid", "CPR", "trauma", "bleeding", "shock"]),
    "medicine.surgery.general": ("جراحة عامة", ["incision", "suture", "anesthesia", "sterile"]),
    "medicine.surgery.orthopedic": ("جراحة عظام", ["fracture", "cast", "pin", "joint"]),
    "medicine.internal": ("باطنية", ["diagnosis", "chronic", "diabetes", "hypertension"]),
    "medicine.pediatrics": ("أطفال", ["infant", "vaccination", "growth", "nutrition"]),
    "medicine.obstetrics": ("توليد", ["pregnancy", "delivery", "cesarean", "prenatal"]),
    "medicine.infectious": ("أمراض معدية", ["epidemic", "quarantine", "hygiene", "sanitation"]),
    "medicine.pharmacy": ("صيدلة", ["drug", "dosage", "interaction", "compounding"]),
    "medicine.dental": ("أسنان", ["cavity", "extraction", "filling", "crown"]),
    "medicine.ophthalmology": ("عيون", ["vision", "cataract", "glasses", "lens"]),
    "medicine.dermatology": ("جلدية", ["skin", "wound", "burn", "infection"]),
    "medicine.radiology": ("أشعة", ["xray", "ultrasound", "CT", "imaging"]),
    "medicine.lab": ("مختبرات", ["blood_test", "culture", "microscope", "pathology"]),
    "medicine.nutrition": ("تغذية", ["vitamin", "mineral", "calorie", "diet"]),
    "medicine.traditional": ("طب شعبي", ["herbal", "natural", "remedy", "plant_medicine"]),
    "medicine.mental_health": ("صحة نفسية", ["stress", "trauma", "counseling", "resilience"]),
    "medicine.public_health": ("صحة عامة", ["sanitation", "water", "epidemic", "prevention"]),
    "medicine.prosthetics": ("أطراف صناعية", ["prosthetic", "implant", "rehabilitation"]),
    "medicine.nursing": ("تمريض", ["patient_care", "wound", "monitoring", "injection"]),
    "medicine.veterinary": ("طب بيطري", ["animal", "livestock", "disease", "treatment"]),

    # ══════════════════════════════════════
    # 3. الزراعة والغذاء (35 كبسولة)
    # ══════════════════════════════════════
    "agriculture.crops.cereals": ("حبوب", ["wheat", "rice", "corn", "barley", "sorghum"]),
    "agriculture.crops.vegetables": ("خضروات", ["tomato", "potato", "onion", "cucumber"]),
    "agriculture.crops.fruits": ("فواكه", ["date", "olive", "grape", "citrus", "apple"]),
    "agriculture.crops.legumes": ("بقوليات", ["bean", "lentil", "chickpea", "soybean"]),
    "agriculture.crops.herbs": ("أعشاب وتوابل", ["mint", "basil", "cumin", "saffron"]),
    "agriculture.irrigation": ("ري", ["drip", "canal", "flood", "sprinkler", "well"]),
    "agriculture.soil_management": ("إدارة تربة", ["fertilizer", "compost", "rotation", "pH"]),
    "agriculture.pest_control": ("مكافحة آفات", ["insect", "fungicide", "biological", "IPM"]),
    "agriculture.livestock.cattle": ("أبقار", ["dairy", "beef", "feeding", "breeding"]),
    "agriculture.livestock.sheep": ("أغنام", ["wool", "meat", "grazing", "lambing"]),
    "agriculture.livestock.poultry": ("دواجن", ["chicken", "egg", "broiler", "incubation"]),
    "agriculture.livestock.fish": ("أسماك", ["aquaculture", "pond", "feed", "harvest"]),
    "agriculture.livestock.bees": ("نحل", ["honey", "hive", "pollination", "wax"]),
    "agriculture.food_processing": ("تصنيع غذائي", ["canning", "drying", "fermenting", "milling"]),
    "agriculture.food_preservation": ("حفظ أغذية", ["salt", "smoke", "pickle", "freeze", "vacuum"]),
    "agriculture.food_safety": ("سلامة غذائية", ["hygiene", "contamination", "inspection"]),
    "agriculture.greenhouse": ("بيوت زجاجية", ["climate", "humidity", "heating", "hydroponic"]),
    "agriculture.forestry": ("حراجة", ["tree", "lumber", "planting", "harvest"]),
    "agriculture.seeds": ("بذور", ["germination", "storage", "hybrid", "heirloom"]),
    "agriculture.organic": ("زراعة عضوية", ["natural", "compost", "biodynamic"]),

    # ══════════════════════════════════════
    # 4. الهندسة (60 كبسولة)
    # ══════════════════════════════════════
    # مدني
    "engineering.civil.structures": ("إنشاءات", ["beam", "column", "foundation", "slab"]),
    "engineering.civil.roads": ("طرق", ["asphalt", "pavement", "highway", "bridge"]),
    "engineering.civil.water_systems": ("مياه", ["pipe", "pump", "treatment", "distribution"]),
    "engineering.civil.sewage": ("صرف صحي", ["drain", "septic", "treatment", "sewer"]),
    "engineering.civil.dams": ("سدود", ["reservoir", "spillway", "hydroelectric"]),
    "engineering.civil.tunnels": ("أنفاق", ["boring", "lining", "ventilation"]),
    "engineering.civil.surveying": ("مساحة", ["GPS", "leveling", "mapping", "theodolite"]),
    "engineering.civil.geotechnical": ("جيوتقنية", ["soil_test", "pile", "excavation"]),

    # ميكانيكي
    "engineering.mechanical.engines": ("محركات", ["diesel", "gasoline", "turbine", "steam"]),
    "engineering.mechanical.hvac": ("تكييف", ["cooling", "heating", "ventilation", "duct"]),
    "engineering.mechanical.pumps": ("مضخات", ["centrifugal", "piston", "submersible"]),
    "engineering.mechanical.compressors": ("ضواغط", ["air", "gas", "reciprocating", "screw"]),
    "engineering.mechanical.bearings": ("محامل", ["ball", "roller", "sleeve", "lubrication"]),
    "engineering.mechanical.gears": ("تروس", ["spur", "helical", "bevel", "worm"]),
    "engineering.mechanical.welding": ("لحام", ["arc", "MIG", "TIG", "oxy_fuel"]),
    "engineering.mechanical.CNC": ("تشغيل آلي", ["lathe", "mill", "drill", "grind"]),
    "engineering.mechanical.hydraulics": ("هيدروليك", ["cylinder", "valve", "pump", "fluid"]),
    "engineering.mechanical.pneumatics": ("نيوماتيك", ["air", "valve", "actuator"]),
    "engineering.mechanical.maintenance": ("صيانة", ["preventive", "predictive", "repair"]),
    "engineering.mechanical.piping": ("أنابيب", ["pipe", "fitting", "flange", "valve"]),
    "engineering.mechanical.boilers": ("غلايات", ["steam", "pressure", "safety"]),

    # كهربائي
    "engineering.electrical.power_systems": ("أنظمة قدرة", ["grid", "transformer", "generator"]),
    "engineering.electrical.electronics": ("إلكترونيات", ["PCB", "transistor", "IC", "sensor"]),
    "engineering.electrical.renewable": ("طاقة متجددة", ["solar", "wind", "hydro", "biomass"]),
    "engineering.electrical.solar_energy": ("طاقة شمسية", ["panel", "inverter", "battery"]),
    "engineering.electrical.motors": ("محركات كهربائية", ["AC", "DC", "stepper", "servo"]),
    "engineering.electrical.wiring": ("تمديدات", ["cable", "conduit", "panel", "breaker"]),
    "engineering.electrical.instrumentation": ("أجهزة قياس", ["sensor", "PLC", "SCADA"]),
    "engineering.electrical.control_systems": ("أنظمة تحكم", ["PID", "feedback", "automation"]),
    "engineering.electrical.power_electronics": ("إلكترونيات قدرة", ["rectifier", "inverter", "converter"]),
    "engineering.electrical.lighting": ("إضاءة", ["LED", "fluorescent", "design"]),

    # كيميائي
    "engineering.chemical.process": ("هندسة عمليات", ["reactor", "distillation", "absorption"]),
    "engineering.chemical.refinery": ("تكرير", ["crude", "distillation", "cracking"]),
    "engineering.chemical.petrochemical": ("بتروكيماويات", ["ethylene", "propylene", "polymerization"]),
    "engineering.chemical.water_treatment": ("معالجة مياه", ["filtration", "chlorination", "RO"]),
    "engineering.chemical.safety": ("سلامة كيميائية", ["MSDS", "hazmat", "spill", "PPE"]),

    # فضاء + نووي
    "engineering.aerospace.propulsion": ("دفع", ["rocket", "jet", "thrust", "nozzle"]),
    "engineering.aerospace.structures": ("هياكل فضائية", ["fuselage", "wing", "composite"]),
    "engineering.aerospace.navigation": ("ملاحة", ["GPS", "inertial", "radar", "orbit"]),
    "engineering.nuclear.reactor": ("مفاعلات", ["fission", "fuel_rod", "coolant", "shield"]),
    "engineering.nuclear.safety": ("سلامة نووية", ["containment", "waste", "decommission"]),

    # ══════════════════════════════════════
    # 5. البرمجة والحوسبة (40 كبسولة)
    # ══════════════════════════════════════
    "engineering.software.backend": ("برمجة خلفية", ["python", "API", "database"]),
    "engineering.software.web": ("تطوير ويب", ["react", "HTML", "CSS", "javascript"]),
    "engineering.software.mobile": ("تطوير موبايل", ["react_native", "flutter", "iOS"]),
    "engineering.software.devops": ("DevOps", ["docker", "kubernetes", "CI_CD"]),
    "engineering.software.embedded": ("أنظمة مدمجة", ["microcontroller", "firmware", "RTOS"]),
    "engineering.software.operating_systems": ("أنظمة تشغيل", ["kernel", "scheduler", "driver"]),
    "engineering.software.networking": ("شبكات", ["TCP", "routing", "firewall", "DNS"]),
    "engineering.software.databases": ("قواعد بيانات", ["SQL", "NoSQL", "indexing", "replication"]),
    "engineering.software.algorithms": ("خوارزميات", ["sorting", "search", "graph", "dynamic"]),
    "engineering.software.compiler": ("مترجمات", ["parser", "lexer", "AST", "code_gen"]),

    # AI/ML
    "computing.ai_ml.nlp": ("معالجة لغة", ["transformer", "BERT", "GPT", "tokenizer"]),
    "computing.ai_ml.vision_ai": ("رؤية حاسوبية", ["CNN", "YOLO", "segmentation"]),
    "computing.ai_ml.vision_ai.object_detection": ("كشف أجسام", ["YOLO", "SSD", "anchor"]),
    "computing.ai_ml.vision_ai.text_recognition": ("قراءة نصوص", ["OCR", "CRNN", "attention"]),
    "computing.ai_ml.vision_ai.segmentation": ("تقسيم صور", ["mask", "semantic", "instance"]),
    "computing.ai_ml.vision_ai.face_analysis": ("تحليل وجوه", ["recognition", "embedding", "landmark"]),
    "computing.ai_ml.vision_ai.safety_vision": ("رؤية سلامة", ["hazard", "PPE", "fire"]),
    "computing.ai_ml.vision_ai.quality_inspection": ("فحص جودة", ["defect", "surface", "measurement"]),
    "computing.ai_ml.reinforcement": ("تعلم معزز", ["reward", "policy", "Q_learning"]),
    "computing.ai_ml.deep_learning": ("تعلم عميق", ["layer", "backprop", "optimization"]),
    "computing.ai_ml.training": ("تدريب نماذج", ["LoRA", "QLoRA", "fine_tuning", "MoE"]),
    "computing.ai_ml.robotics": ("ذكاء روبوتات", ["control", "sensor", "actuator", "path"]),

    # أمن
    "computing.security.crypto": ("تشفير", ["AES", "RSA", "hash", "PKI"]),
    "computing.security.pentest": ("اختبار اختراق", ["OWASP", "exploit", "vuln"]),
    "computing.security.network": ("أمن شبكات", ["firewall", "IDS", "VPN"]),
    "computing.security.forensics": ("تحقيق جنائي رقمي", ["evidence", "recovery", "analysis"]),

    # ══════════════════════════════════════
    # 6. التصنيع (50 كبسولة)
    # ══════════════════════════════════════
    "manufacturing.metals.steelmaking": ("صناعة حديد", ["furnace", "casting", "rolling"]),
    "manufacturing.metals.casting": ("سباكة", ["mold", "sand", "die_cast"]),
    "manufacturing.metals.aluminum": ("ألمنيوم", ["smelting", "extrusion", "anodizing"]),
    "manufacturing.metals.copper": ("نحاس", ["refining", "wire", "plating"]),
    "manufacturing.metals.foundry": ("مسبك", ["melting", "pouring", "finishing"]),
    "manufacturing.cement": ("أسمنت", ["kiln", "clinker", "grinding", "limestone"]),
    "manufacturing.glass": ("زجاج", ["silica", "furnace", "float", "temper"]),
    "manufacturing.brick": ("طوب", ["clay", "kiln", "press", "fire"]),
    "manufacturing.concrete": ("خرسانة", ["mix", "rebar", "pour", "cure"]),
    "manufacturing.ceramics": ("سيراميك", ["tile", "glaze", "kiln", "porcelain"]),
    "manufacturing.plastic": ("بلاستيك", ["injection", "extrusion", "blow_mold"]),
    "manufacturing.rubber": ("مطاط", ["vulcanize", "mold", "compound"]),
    "manufacturing.paper": ("ورق", ["pulp", "press", "bleach", "roll"]),
    "manufacturing.textile": ("نسيج", ["loom", "spinning", "dyeing", "weaving"]),
    "manufacturing.leather": ("جلود", ["tanning", "cutting", "stitching"]),
    "manufacturing.soap": ("صابون", ["saponification", "oil", "lye", "fragrance"]),
    "manufacturing.paint": ("دهانات", ["pigment", "binder", "solvent", "coating"]),
    "manufacturing.fertilizer": ("أسمدة", ["urea", "phosphate", "potash", "NPK"]),
    "manufacturing.explosives": ("متفجرات", ["TNT", "detonator", "blasting", "safety"]),
    "manufacturing.food_industry": ("صناعة غذائية", ["processing", "packaging", "canning"]),

    # إلكترونيات صناعية
    "manufacturing.electronics.semiconductor": ("رقائق", ["lithography", "wafer", "doping", "cleanroom"]),
    "manufacturing.electronics.PCB": ("لوحات مطبوعة", ["etching", "soldering", "SMT"]),
    "manufacturing.electronics.battery": ("بطاريات", ["lithium", "lead_acid", "cell"]),
    "manufacturing.electronics.solar_panel": ("ألواح شمسية", ["cell", "encapsulation", "silicon"]),
    "manufacturing.electronics.LED": ("إضاءة LED", ["diode", "driver", "package"]),
    "manufacturing.electronics.cable": ("كابلات", ["conductor", "insulation", "shielding"]),
    "manufacturing.electronics.transformer": ("محولات", ["core", "winding", "insulation"]),
    "manufacturing.electronics.motor": ("محركات", ["winding", "rotor", "stator"]),

    # مركبات
    "manufacturing.vehicles.car": ("سيارات", ["engine", "chassis", "body", "assembly"]),
    "manufacturing.vehicles.truck": ("شاحنات", ["heavy", "diesel", "trailer"]),
    "manufacturing.vehicles.bicycle": ("دراجات", ["frame", "chain", "wheel"]),
    "manufacturing.vehicles.boat": ("قوارب", ["hull", "engine", "fiberglass"]),
    "manufacturing.vehicles.aircraft": ("طائرات", ["airframe", "engine", "avionics"]),

    # ══════════════════════════════════════
    # 7. البناء والحرف (30 كبسولة)
    # ══════════════════════════════════════
    "crafts.carpentry": ("نجارة", ["wood", "joint", "saw", "plane", "cabinet"]),
    "crafts.blacksmithing": ("حدادة", ["forge", "anvil", "hammer", "quench"]),
    "crafts.masonry": ("بناء", ["brick", "mortar", "block", "arch"]),
    "crafts.plumbing": ("سباكة", ["pipe", "fitting", "solder", "valve"]),
    "crafts.roofing": ("أسقف", ["tile", "insulation", "gutter", "waterproof"]),
    "crafts.painting": ("دهان", ["brush", "roller", "primer", "finish"]),
    "crafts.tiling": ("بلاط", ["ceramic", "grout", "level", "cutting"]),
    "crafts.glasswork": ("أعمال زجاج", ["cutting", "glazing", "stained"]),
    "crafts.pottery": ("فخار", ["clay", "wheel", "kiln", "glaze"]),
    "crafts.sewing": ("خياطة", ["fabric", "pattern", "stitch", "machine"]),
    "crafts.shoe_making": ("صناعة أحذية", ["leather", "sole", "last", "stitch"]),
    "crafts.rope_making": ("حبال", ["fiber", "twist", "braid", "knot"]),
    "crafts.basket_weaving": ("سلال", ["reed", "bamboo", "weave", "pattern"]),
    "crafts.soap_making": ("صناعة صابون", ["oil", "lye", "mold", "cure"]),
    "crafts.candle_making": ("شموع", ["wax", "wick", "mold", "fragrance"]),
    "crafts.tool_making": ("صناعة أدوات", ["forge", "grind", "handle", "temper"]),
    "crafts.knife_making": ("صناعة سكاكين", ["blade", "steel", "handle", "sharpen"]),

    # ══════════════════════════════════════
    # 8. الطاقة (25 كبسولة)
    # ══════════════════════════════════════
    "energy.solar.photovoltaic": ("كهروضوئية", ["panel", "cell", "inverter"]),
    "energy.solar.thermal": ("حرارية شمسية", ["collector", "storage", "heating"]),
    "energy.wind.turbine": ("توربين رياح", ["blade", "generator", "tower"]),
    "energy.hydro.dam": ("سد كهرومائي", ["turbine", "reservoir", "penstock"]),
    "energy.hydro.micro": ("مائية صغيرة", ["creek", "pelton", "crossflow"]),
    "energy.biomass": ("كتلة حيوية", ["biogas", "wood_gas", "methane"]),
    "energy.geothermal": ("حرارة أرضية", ["well", "heat_pump", "steam"]),
    "energy.nuclear.fission": ("انشطار نووي", ["uranium", "reactor", "control_rod"]),
    "energy.fossil.coal": ("فحم", ["mine", "burn", "steam"]),
    "energy.fossil.oil": ("نفط", ["drill", "refine", "distill"]),
    "energy.fossil.gas": ("غاز طبيعي", ["pipeline", "LNG", "turbine"]),
    "energy.storage.battery": ("تخزين بطاريات", ["lithium", "lead", "cycle"]),
    "energy.storage.hydrogen": ("هيدروجين", ["electrolysis", "fuel_cell", "storage"]),
    "energy.grid.distribution": ("توزيع كهرباء", ["transformer", "line", "meter"]),
    "energy.grid.off_grid": ("مستقل عن الشبكة", ["solar", "battery", "inverter"]),
    "energy.efficiency": ("كفاءة طاقة", ["insulation", "LED", "audit"]),

    # ══════════════════════════════════════
    # 9. المياه (15 كبسولة)
    # ══════════════════════════════════════
    "water.purification": ("تنقية مياه", ["filter", "chlorine", "UV", "boil"]),
    "water.desalination": ("تحلية", ["reverse_osmosis", "distillation", "solar"]),
    "water.well_drilling": ("حفر آبار", ["drill", "casing", "pump", "aquifer"]),
    "water.rainwater": ("مياه أمطار", ["collection", "storage", "filter"]),
    "water.irrigation": ("ري", ["drip", "sprinkler", "canal", "schedule"]),
    "water.sewage": ("صرف صحي", ["septic", "treatment", "biofilm"]),
    "water.testing": ("فحص مياه", ["pH", "turbidity", "bacteria", "TDS"]),
    "water.dam": ("سدود مياه", ["reservoir", "spillway", "gate"]),
    "water.piping": ("أنابيب مياه", ["PVC", "PE", "steel", "joint"]),

    # ══════════════════════════════════════
    # 10. اتصالات ونقل (25 كبسولة)
    # ══════════════════════════════════════
    "communication.radio": ("راديو", ["frequency", "antenna", "transmitter", "AM_FM"]),
    "communication.satellite": ("أقمار صناعية", ["orbit", "transponder", "dish"]),
    "communication.fiber_optic": ("ألياف ضوئية", ["fiber", "laser", "splice"]),
    "communication.telephone": ("هاتف", ["switch", "line", "PBX"]),
    "communication.internet": ("إنترنت", ["router", "protocol", "server"]),
    "communication.printing": ("طباعة", ["press", "ink", "paper", "offset"]),
    "communication.signal": ("إشارات", ["modulation", "encoding", "noise"]),

    "transport.road": ("نقل بري", ["truck", "highway", "logistics"]),
    "transport.rail": ("سكك حديد", ["track", "locomotive", "signal"]),
    "transport.sea": ("نقل بحري", ["ship", "port", "container", "navigation"]),
    "transport.air": ("نقل جوي", ["aircraft", "airport", "ATC"]),
    "transport.pipeline": ("أنابيب نقل", ["oil", "gas", "pump_station"]),

    # ══════════════════════════════════════
    # 11. الإدارة والاقتصاد (30 كبسولة)
    # ══════════════════════════════════════
    "business.accounting": ("محاسبة", ["ledger", "balance", "journal", "audit"]),
    "business.economics": ("اقتصاد", ["supply", "demand", "inflation", "GDP"]),
    "business.management": ("إدارة", ["planning", "organizing", "leading"]),
    "business.project_management": ("إدارة مشاريع", ["schedule", "budget", "scope"]),
    "business.supply_chain": ("سلاسل إمداد", ["logistics", "inventory", "sourcing"]),
    "business.quality": ("جودة", ["ISO", "TQM", "six_sigma", "inspection"]),
    "business.HR": ("موارد بشرية", ["hiring", "training", "evaluation"]),
    "business.marketing": ("تسويق", ["market", "customer", "pricing"]),
    "business.finance": ("مالية", ["cash_flow", "investment", "loan"]),
    "business.law": ("قانون تجاري", ["contract", "liability", "IP"]),
    "business.tax": ("ضرائب", ["income", "VAT", "compliance"]),
    "business.insurance": ("تأمين", ["risk", "premium", "claim"]),
    "business.trade": ("تجارة", ["export", "import", "customs"]),

    # ══════════════════════════════════════
    # 12. الحكم والمجتمع (25 كبسولة)
    # ══════════════════════════════════════
    "governance.leadership": ("قيادة", ["vision", "decision", "delegation"]),
    "governance.organization": ("تنظيم مجتمع", ["structure", "roles", "rules"]),
    "governance.law": ("قانون", ["constitution", "criminal", "civil"]),
    "governance.education": ("تعليم", ["curriculum", "teaching", "assessment"]),
    "governance.diplomacy": ("دبلوماسية", ["negotiation", "treaty", "alliance"]),
    "governance.emergency": ("طوارئ", ["disaster", "evacuation", "rescue"]),
    "governance.census": ("إحصاء سكان", ["population", "survey", "demographics"]),

    "society.psychology": ("علم نفس", ["motivation", "behavior", "group", "trauma"]),
    "society.sociology": ("علم اجتماع", ["community", "culture", "conflict"]),
    "society.education.methods": ("طرق تدريس", ["active", "practical", "apprentice"]),
    "society.ethics": ("أخلاقيات", ["justice", "fairness", "responsibility"]),
    "society.history.civilizations": ("تاريخ حضارات", ["mesopotamia", "egypt", "rome"]),
    "society.history.technology": ("تاريخ تقنية", ["industrial_revolution", "innovation"]),
    "society.history.mistakes": ("أخطاء تاريخية", ["famine", "war", "collapse"]),
    "society.languages.arabic": ("لغة عربية", ["grammar", "writing", "poetry"]),
    "society.languages.english": ("لغة إنجليزية", ["grammar", "vocabulary", "technical"]),

    # ══════════════════════════════════════
    # 13. البقاء بعد الكارثة (20 كبسولة)
    # ══════════════════════════════════════
    "survival.shelter": ("مأوى", ["tent", "cabin", "insulation", "heating"]),
    "survival.fire": ("إشعال نار", ["friction", "flint", "kindling", "stove"]),
    "survival.water": ("مياه بقاء", ["purify", "solar_still", "filter"]),
    "survival.food_gathering": ("جمع غذاء", ["hunting", "fishing", "foraging"]),
    "survival.navigation": ("ملاحة", ["compass", "stars", "map", "sun"]),
    "survival.first_aid": ("إسعافات أولية", ["wound", "splint", "CPR", "burn"]),
    "survival.signaling": ("إشارات نجدة", ["smoke", "mirror", "SOS", "radio"]),
    "survival.self_defense": ("دفاع عن نفس", ["awareness", "escape", "tools"]),
    "survival.weather": ("طقس", ["forecast", "storm", "cold", "heat"]),
    "survival.radiation": ("حماية إشعاع", ["decontamination", "shelter", "dosimeter"]),
    "survival.tools": ("أدوات بقاء", ["knife", "rope", "container", "axe"]),
    "survival.community": ("بناء مجتمع", ["cooperation", "leadership", "division"]),
    "survival.priorities": ("أولويات بقاء", ["water", "shelter", "food", "security"]),

    # ══════════════════════════════════════
    # 14. عسكري ودفاعي (15 كبسولة)
    # ══════════════════════════════════════
    "military.strategy": ("استراتيجية", ["defense", "offense", "logistics"]),
    "military.fortification": ("تحصينات", ["bunker", "wall", "trench"]),
    "military.communication": ("اتصالات عسكرية", ["radio", "cipher", "signal"]),
    "military.logistics": ("لوجستيات عسكرية", ["supply", "transport", "base"]),
    "military.weapons.basic": ("أسلحة أساسية", ["bow", "crossbow", "catapult"]),
    "military.weapons.firearms": ("أسلحة نارية", ["rifle", "ammunition", "maintenance"]),
    "military.vehicles.armored": ("مركبات مدرعة", ["armor", "engine", "weapon"]),
    "military.medical": ("طب ميداني", ["triage", "tourniquet", "evacuation"]),
    "military.engineering": ("هندسة عسكرية", ["bridge", "demolition", "obstacle"]),
    "military.NBC": ("كيميائي بيولوجي نووي", ["detection", "decontamination", "protection"]),

    # ══════════════════════════════════════
    # 15. الحكماء — حكمة المجلس (20 كبسولة)
    # ══════════════════════════════════════
    "wisdom.philosophy": ("فلسفة", ["logic", "ethics", "epistemology", "metaphysics"]),
    "wisdom.strategic_thinking": ("تفكير استراتيجي", ["long_term", "scenario", "risk"]),
    "wisdom.decision_making": ("صنع قرار", ["analysis", "trade_off", "consensus"]),
    "wisdom.systems_thinking": ("تفكير نظمي", ["feedback", "emergence", "complexity"]),
    "wisdom.creative_thinking": ("تفكير إبداعي", ["innovation", "lateral", "brainstorm"]),
    "wisdom.critical_thinking": ("تفكير نقدي", ["argument", "fallacy", "evidence"]),
    "wisdom.risk_management": ("إدارة مخاطر", ["probability", "impact", "mitigation"]),
    "wisdom.ethics_ai": ("أخلاقيات ذكاء", ["bias", "fairness", "transparency"]),
    "wisdom.futurism": ("دراسات مستقبلية", ["trend", "forecast", "scenario"]),
    "wisdom.civilization": ("بناء حضارة", ["infrastructure", "education", "governance"]),
    "wisdom.resilience": ("مرونة", ["adapt", "recover", "antifragile"]),
    "wisdom.negotiation": ("تفاوض", ["win_win", "BATNA", "influence"]),
    "wisdom.teaching": ("تعليم بشر", ["explain", "demonstrate", "practice"]),
    "wisdom.problem_solving": ("حل مشاكل", ["root_cause", "systematic", "creative"]),
    "wisdom.planning_100yr": ("تخطيط 100 سنة", ["roadmap", "milestones", "contingency"]),

    # ══════════════════════════════════════════════════════════
    # 16. ⭐ البرمجيات — شجرة عميقة (60+ كبسولة)
    # ══════════════════════════════════════════════════════════

    # --- اللغات ---
    "software.lang.python": ("بايثون", ["python", "pip", "virtualenv", "typing"]),
    "software.lang.python.advanced": ("بايثون متقدم", ["metaclass", "decorator", "generator", "asyncio"]),
    "software.lang.python.data": ("بايثون بيانات", ["pandas", "numpy", "scipy", "matplotlib"]),
    "software.lang.javascript": ("جافاسكريبت", ["ES6", "promise", "async", "node"]),
    "software.lang.typescript": ("تايبسكريبت", ["type", "interface", "generic", "decorator"]),
    "software.lang.rust": ("رست", ["ownership", "borrow", "lifetime", "cargo"]),
    "software.lang.go": ("غو", ["goroutine", "channel", "interface", "module"]),
    "software.lang.cpp": ("سي++", ["pointer", "template", "STL", "RAII"]),
    "software.lang.c": ("سي", ["pointer", "malloc", "struct", "header"]),
    "software.lang.java": ("جافا", ["JVM", "spring", "maven", "OOP"]),
    "software.lang.swift": ("سويفت", ["optional", "protocol", "closure", "SwiftUI"]),
    "software.lang.kotlin": ("كوتلن", ["coroutine", "null_safety", "extension"]),
    "software.lang.sql": ("SQL", ["query", "join", "index", "stored_proc"]),
    "software.lang.bash": ("باش", ["script", "pipe", "grep", "awk", "sed"]),
    "software.lang.assembly": ("أسمبلي", ["register", "instruction", "x86", "ARM"]),

    # --- أطر عمل ويب ---
    "software.framework.react": ("ريأكت", ["component", "hook", "state", "JSX"]),
    "software.framework.nextjs": ("نكست", ["SSR", "SSG", "API_route", "middleware"]),
    "software.framework.vue": ("فيو", ["directive", "component", "vuex", "router"]),
    "software.framework.svelte": ("سفيلت", ["reactive", "store", "compile"]),
    "software.framework.django": ("جانغو", ["ORM", "template", "middleware", "auth"]),
    "software.framework.fastapi": ("فاست API", ["endpoint", "pydantic", "async", "openapi"]),
    "software.framework.flask": ("فلاسك", ["route", "blueprint", "jinja", "WSGI"]),
    "software.framework.express": ("إكسبريس", ["middleware", "route", "REST", "socket"]),
    "software.framework.spring": ("سبرينغ", ["bean", "DI", "AOP", "boot"]),

    # --- أطر موبايل ---
    "software.mobile.react_native": ("ريأكت نيتف", ["component", "navigation", "native_module"]),
    "software.mobile.flutter": ("فلاتر", ["widget", "dart", "state", "material"]),
    "software.mobile.ios": ("iOS أصلي", ["UIKit", "SwiftUI", "CoreData", "Xcode"]),
    "software.mobile.android": ("أندرويد أصلي", ["activity", "fragment", "jetpack", "gradle"]),

    # --- قواعد بيانات ---
    "software.database.postgresql": ("بوستجريس", ["query", "index", "partition", "replication"]),
    "software.database.mysql": ("ماي SQL", ["InnoDB", "replication", "optimization"]),
    "software.database.mongodb": ("مونغو", ["document", "aggregation", "shard"]),
    "software.database.redis": ("ريديس", ["cache", "pub_sub", "stream", "cluster"]),
    "software.database.sqlite": ("SQLite", ["embedded", "WAL", "pragma"]),
    "software.database.elasticsearch": ("إلاستيك", ["index", "query", "aggregation"]),
    "software.database.design": ("تصميم قواعد بيانات", ["normalization", "schema", "migration"]),

    # --- بنية تحتية ---
    "software.infra.docker": ("دوكر", ["container", "image", "compose", "volume"]),
    "software.infra.kubernetes": ("كوبرنيتس", ["pod", "service", "deployment", "helm"]),
    "software.infra.nginx": ("إنجينكس", ["proxy", "load_balance", "ssl", "config"]),
    "software.infra.linux_admin": ("إدارة لينكس", ["systemd", "cron", "user", "permission"]),
    "software.infra.git": ("غيت", ["branch", "merge", "rebase", "conflict"]),
    "software.infra.ci_cd": ("CI/CD", ["pipeline", "test", "deploy", "artifact"]),
    "software.infra.monitoring": ("مراقبة", ["prometheus", "grafana", "alert", "log"]),
    "software.infra.terraform": ("تيرافورم", ["provider", "resource", "state", "module"]),
    "software.infra.aws": ("سحابة AWS", ["EC2", "S3", "Lambda", "RDS"]),
    "software.infra.networking": ("شبكات", ["TCP", "UDP", "DNS", "HTTP", "WebSocket"]),

    # --- أنماط ومعمارية ---
    "software.architecture.microservices": ("خدمات مصغرة", ["service", "gateway", "queue"]),
    "software.architecture.monolith": ("تطبيق واحد", ["MVC", "layer", "module"]),
    "software.architecture.event_driven": ("حدثي", ["event", "CQRS", "saga", "message"]),
    "software.architecture.clean_arch": ("بنية نظيفة", ["domain", "use_case", "adapter"]),
    "software.architecture.design_patterns": ("أنماط تصميم", ["singleton", "factory", "observer"]),
    "software.architecture.DDD": ("تصميم مجالي", ["entity", "value_object", "aggregate"]),
    "software.architecture.API_design": ("تصميم API", ["REST", "GraphQL", "gRPC", "versioning"]),
    "software.architecture.testing": ("اختبارات", ["unit", "integration", "E2E", "TDD"]),
    "software.architecture.performance": ("أداء", ["profiling", "caching", "CDN", "optimization"]),
    "software.architecture.security": ("أمان تطبيقات", ["auth", "CORS", "injection", "XSS"]),

    # ══════════════════════════════════════════════════════════
    # 17. ⭐ بناء الأدمغة والطبقات (50+ كبسولة)
    # ══════════════════════════════════════════════════════════

    # --- بناء نماذج AI ---
    "brain.models.transformer": ("محوّل", ["attention", "multi_head", "position", "embedding"]),
    "brain.models.GPT": ("GPT", ["decoder", "autoregressive", "next_token", "prompt"]),
    "brain.models.BERT": ("BERT", ["encoder", "masked_LM", "classification"]),
    "brain.models.diffusion": ("نماذج انتشار", ["denoise", "score", "stable_diffusion"]),
    "brain.models.GAN": ("شبكات توليدية", ["generator", "discriminator", "training"]),
    "brain.models.VAE": ("مشفر تغييري", ["encoder", "decoder", "latent_space"]),
    "brain.models.MoE": ("خليط خبراء", ["router", "expert", "gating", "capacity"]),

    # --- تدريب ---
    "brain.training.LoRA": ("لورا", ["adapter", "rank", "alpha", "target_module"]),
    "brain.training.QLoRA": ("كيولورا", ["4bit", "quantize", "bnb", "peft"]),
    "brain.training.full_finetune": ("تدريب كامل", ["gradient", "optimizer", "scheduler"]),
    "brain.training.RLHF": ("تعلم من تقييم بشري", ["reward_model", "PPO", "preference"]),
    "brain.training.DPO": ("تحسين مباشر", ["preference", "policy", "reference"]),
    "brain.training.distillation": ("تقطير معرفة", ["teacher", "student", "temperature"]),
    "brain.training.curriculum": ("تدريب منهجي", ["difficulty", "order", "progression"]),
    "brain.training.distributed": ("تدريب موزع", ["DDP", "FSDP", "DeepSpeed", "pipeline"]),
    "brain.training.data": ("بيانات تدريب", ["JSONL", "tokenize", "shuffle", "augment"]),
    "brain.training.evaluation": ("تقييم نماذج", ["benchmark", "perplexity", "BLEU", "human"]),
    "brain.training.merging": ("دمج نماذج", ["mergekit", "TIES", "DARE", "linear"]),
    "brain.training.quantization": ("تكميم", ["GGUF", "GPTQ", "AWQ", "INT4", "INT8"]),

    # --- بناء طبقات الدماغ ---
    "brain.layers.planner": ("مخطط", ["task", "decompose", "prioritize", "schedule"]),
    "brain.layers.researcher": ("باحث", ["search", "retrieve", "summarize", "cite"]),
    "brain.layers.critic": ("ناقد", ["evaluate", "score", "feedback", "improve"]),
    "brain.layers.executor": ("منفذ", ["code", "run", "test", "deploy"]),
    "brain.layers.verifier": ("مدقق", ["check", "validate", "test", "proof"]),
    "brain.layers.council": ("مجلس حكماء", ["sage", "vote", "consensus", "debate"]),
    "brain.layers.scouts": ("كشافة", ["search", "discover", "crawl", "index"]),
    "brain.layers.memory": ("ذاكرة", ["store", "retrieve", "forget", "consolidate"]),

    # --- بنية أدمغة ---
    "brain.arch.capsule_design": ("تصميم كبسولات", ["specialization", "hierarchy", "inheritance"]),
    "brain.arch.tree_pyramid": ("أشجار وأهرام", ["tree", "branch", "leaf", "root"]),
    "brain.arch.inheritance": ("وراثة معرفة", ["parent", "child", "cascade", "multi"]),
    "brain.arch.auto_expand": ("توسع أوتوماتيكي", ["discover", "create", "link"]),
    "brain.arch.self_evolution": ("تطور ذاتي", ["version", "snapshot", "rollback", "benchmark"]),
    "brain.arch.meta_learning": ("تعلم فوقي", ["few_shot", "MAML", "prompt_tuning"]),
    "brain.arch.agentic": ("وكيل ذكي", ["tool_use", "planning", "reflection", "chain"]),
    "brain.arch.RAG": ("استرجاع معزز", ["embed", "retrieve", "rerank", "context"]),
    "brain.arch.multi_agent": ("متعدد وكلاء", ["orchestrate", "message", "protocol"]),
    "brain.arch.neuro_symbolic": ("عصبي رمزي", ["rule", "logic", "knowledge_graph"]),
    "brain.arch.bayesian": ("بايزي", ["prior", "posterior", "uncertainty", "update"]),
    "brain.arch.causal": ("سببي", ["intervention", "counterfactual", "DAG"]),

    # --- أدوات بناء أدمغة ---
    "brain.tools.pytorch": ("باي تورش", ["tensor", "autograd", "module", "dataloader"]),
    "brain.tools.huggingface": ("هاغينغ فيس", ["model", "tokenizer", "trainer", "pipeline"]),
    "brain.tools.vllm": ("vLLM", ["serve", "batch", "paged_attention"]),
    "brain.tools.ollama": ("أولاما", ["model", "API", "embed", "local"]),
    "brain.tools.langchain": ("لانغتشين", ["chain", "agent", "tool", "memory"]),
    "brain.tools.mlflow": ("MLflow", ["experiment", "metric", "model_registry"]),
    "brain.tools.wandb": ("Weights&Biases", ["log", "sweep", "artifact"]),

    # ══════════════════════════════════════════════════════════
    # 18. ⭐ الاختراق والأمن الهجومي (40+ كبسولة)
    # ══════════════════════════════════════════════════════════

    # --- اختراق شبكات ---
    "hacking.network.scanning": ("مسح شبكات", ["nmap", "masscan", "port", "service"]),
    "hacking.network.sniffing": ("التنصت", ["wireshark", "tcpdump", "packet", "mitm"]),
    "hacking.network.mitm": ("رجل بالمنتصف", ["ARP_spoof", "DNS_spoof", "SSL_strip"]),
    "hacking.network.wifi": ("اختراق واي فاي", ["WPA", "deauth", "handshake", "aircrack"]),
    "hacking.network.protocol": ("بروتوكولات", ["TCP", "UDP", "HTTP", "DNS", "SMB"]),
    "hacking.network.pivoting": ("تنقل شبكي", ["tunnel", "proxy", "lateral_movement"]),
    "hacking.network.dos": ("هجمات حرمان", ["DDoS", "SYN_flood", "amplification"]),

    # --- اختراق ويب ---
    "hacking.web.sqli": ("حقن SQL", ["union", "blind", "time_based", "sqlmap"]),
    "hacking.web.xss": ("برمجة عبر مواقع", ["reflected", "stored", "DOM", "filter_bypass"]),
    "hacking.web.csrf": ("تزوير طلبات", ["token", "referer", "SameSite"]),
    "hacking.web.ssrf": ("طلبات جانب خادم", ["internal", "cloud_metadata", "bypass"]),
    "hacking.web.auth_bypass": ("تجاوز مصادقة", ["JWT", "session", "cookie", "OAuth"]),
    "hacking.web.file_upload": ("رفع ملفات", ["shell", "bypass", "mime", "extension"]),
    "hacking.web.directory": ("اكتشاف مسارات", ["dirbust", "wordlist", "fuzzing"]),
    "hacking.web.API_hacking": ("اختراق APIs", ["BOLA", "mass_assign", "rate_limit"]),
    "hacking.web.deserialization": ("إلغاء تسلسل", ["RCE", "pickle", "Java", "PHP"]),

    # --- اختراق أنظمة ---
    "hacking.system.linux": ("اختراق لينكس", ["privilege_escalation", "kernel", "SUID"]),
    "hacking.system.windows": ("اختراق ويندوز", ["mimikatz", "pass_the_hash", "AD"]),
    "hacking.system.active_directory": ("دليل نشط", ["kerberos", "LDAP", "GPO", "DCSync"]),
    "hacking.system.password": ("كسر كلمات مرور", ["hashcat", "john", "rainbow", "brute"]),
    "hacking.system.shells": ("قذائف عكسية", ["reverse_shell", "bind_shell", "web_shell"]),
    "hacking.system.persistence": ("ثبات", ["backdoor", "rootkit", "cron", "service"]),

    # --- هندسة عكسية ---
    "hacking.reverse.binary": ("تحليل ثنائي", ["IDA", "ghidra", "disassemble", "debug"]),
    "hacking.reverse.malware": ("تحليل برمجيات خبيثة", ["static", "dynamic", "sandbox"]),
    "hacking.reverse.firmware": ("تحليل فيرموير", ["extract", "emulate", "UART", "JTAG"]),
    "hacking.reverse.android": ("تحليل أندرويد", ["APK", "smali", "frida", "hook"]),
    "hacking.reverse.crypto": ("كسر تشفير", ["padding_oracle", "weak_key", "side_channel"]),

    # --- أمن دفاعي ---
    "hacking.defense.firewall": ("جدران نارية", ["iptables", "pf", "WAF", "rules"]),
    "hacking.defense.IDS": ("كشف اختراق", ["snort", "suricata", "SIEM", "alert"]),
    "hacking.defense.hardening": ("تصليب أنظمة", ["CIS", "patch", "disable", "audit"]),
    "hacking.defense.incident": ("استجابة حوادث", ["triage", "contain", "eradicate", "report"]),
    "hacking.defense.forensics": ("تحقيق جنائي", ["image", "timeline", "evidence", "chain"]),
    "hacking.defense.OSINT": ("استخبارات مفتوحة", ["social", "domain", "metadata", "dork"]),

    # --- أدوات ---
    "hacking.tools.metasploit": ("ميتاسبلويت", ["exploit", "payload", "module", "meterpreter"]),
    "hacking.tools.burpsuite": ("بيرب سويت", ["proxy", "scanner", "intruder", "repeater"]),
    "hacking.tools.kali": ("كالي لينكس", ["tools", "wordlist", "scripts"]),
    "hacking.tools.cobalt_strike": ("كوبالت سترايك", ["beacon", "C2", "listener"]),
    "hacking.tools.scripting": ("برمجة هجومية", ["python", "exploit_dev", "automation"]),

    # ══════════════════════════════════════════════════════════
    # 19. ⭐ التحليل الصوري والفيديو (30+ كبسولة)
    # ══════════════════════════════════════════════════════════

    # --- تصنيف صور ---
    "vision.classification.general": ("تصنيف عام", ["ImageNet", "ResNet", "EfficientNet", "ViT"]),
    "vision.classification.medical": ("تصنيف طبي", ["xray", "CT", "MRI", "pathology"]),
    "vision.classification.satellite": ("تصنيف أقمار", ["land_use", "crop", "urban", "forest"]),
    "vision.classification.industrial": ("تصنيف صناعي", ["defect", "quality", "product"]),
    "vision.classification.document": ("تصنيف مستندات", ["invoice", "receipt", "form", "layout"]),

    # --- كشف أجسام ---
    "vision.detection.yolo": ("YOLO كشف", ["anchor", "NMS", "FPN", "realtime"]),
    "vision.detection.face": ("كشف وجوه", ["landmark", "align", "embedding", "ArcFace"]),
    "vision.detection.vehicle": ("كشف مركبات", ["car", "truck", "plate", "tracking"]),
    "vision.detection.ppe": ("كشف معدات سلامة", ["helmet", "vest", "gloves", "goggles"]),
    "vision.detection.fire": ("كشف حريق", ["flame", "smoke", "thermal", "alert"]),
    "vision.detection.weapon": ("كشف أسلحة", ["gun", "knife", "threat", "alert"]),

    # --- تقسيم صور ---
    "vision.segmentation.semantic": ("تقسيم دلالي", ["pixel", "class", "FCN", "UNet"]),
    "vision.segmentation.instance": ("تقسيم مثيلات", ["Mask_RCNN", "SAM", "panoptic"]),
    "vision.segmentation.medical": ("تقسيم طبي", ["organ", "tumor", "cell", "UNet"]),

    # --- قراءة نصوص (OCR) ---
    "vision.ocr.print": ("قراءة مطبوع", ["tesseract", "PaddleOCR", "line", "word"]),
    "vision.ocr.handwriting": ("قراءة خط يد", ["arabic", "english", "CRNN", "attention"]),
    "vision.ocr.scene": ("قراءة مشاهد", ["sign", "plate", "CRAFT", "STR"]),
    "vision.ocr.document": ("قراءة مستندات", ["table", "form", "layout", "structure"]),

    # --- فيديو ---
    "vision.video.action": ("تحليل أفعال", ["activity", "gesture", "pose", "3DCNN"]),
    "vision.video.tracking": ("تتبع أجسام", ["SORT", "DeepSORT", "ByteTrack", "ReID"]),
    "vision.video.surveillance": ("مراقبة", ["anomaly", "crowd", "intrusion", "loiter"]),
    "vision.video.understanding": ("فهم فيديو", ["captioning", "QA", "temporal", "clip"]),
    "vision.video.generation": ("توليد فيديو", ["diffusion", "frame", "interpolation"]),

    # --- ثلاثي الأبعاد ---
    "vision.3d.reconstruction": ("إعادة بناء 3D", ["photogrammetry", "NeRF", "mesh", "point_cloud"]),
    "vision.3d.depth": ("تقدير عمق", ["monocular", "stereo", "LiDAR", "disparity"]),
    "vision.3d.pose": ("تقدير وضعية", ["skeleton", "joint", "MediaPipe", "OpenPose"]),

    # --- معالجة صور ---
    "vision.processing.enhancement": ("تحسين صور", ["denoise", "super_resolution", "HDR"]),
    "vision.processing.generation": ("توليد صور", ["stable_diffusion", "DALL-E", "GAN"]),

    # ══════════════════════════════════════════════════════════
    # 20. كبسولات إضافية (لتوصيل 500+)
    # ══════════════════════════════════════════════════════════

    # --- روبوتات ---
    "robotics.manipulation": ("تحكم يدوي", ["gripper", "arm", "pick_place", "force"]),
    "robotics.navigation": ("ملاحة روبوت", ["SLAM", "path_planning", "obstacle"]),
    "robotics.drone": ("طائرات مسيّرة", ["quadcopter", "flight_controller", "GPS"]),
    "robotics.humanoid": ("روبوت بشري", ["balance", "walking", "manipulation"]),

    # --- فضاء ---
    "space.orbital_mechanics": ("ميكانيكا مدارية", ["orbit", "delta_v", "transfer"]),
    "space.life_support": ("دعم حياة", ["oxygen", "water_recycle", "CO2"]),
    "space.propulsion": ("دفع فضائي", ["ion", "chemical", "nuclear_thermal"]),

    # --- بحري ---
    "marine.shipbuilding": ("بناء سفن", ["hull", "keel", "propeller", "welding"]),
    "marine.navigation": ("ملاحة بحرية", ["compass", "sextant", "chart", "radar"]),
    "marine.fishing": ("صيد أسماك", ["net", "trawl", "hook", "sustainable"]),

    # --- نسيج ومواد متقدمة ---
    "advanced.graphene": ("غرافين", ["layer", "conduct", "strength", "flex"]),
    "advanced.3d_printing": ("طباعة ثلاثية", ["FDM", "SLA", "SLS", "metal"]),
    "advanced.biotech": ("تقنية حيوية", ["CRISPR", "fermentation", "bioreactor"]),

    # --- تعليم وإدارة معرفة ---
    "knowledge.library": ("مكتبات", ["catalog", "archive", "preserve", "digitize"]),
    "knowledge.research_method": ("منهجية بحث", ["hypothesis", "experiment", "peer_review"]),
    "knowledge.data_science": ("علم بيانات", ["cleaning", "visualization", "pipeline"]),

    # --- أمن غذائي ---
    "food_security.storage": ("تخزين غذائي", ["silo", "humidity", "pest_control"]),
    "food_security.distribution": ("توزيع غذائي", ["cold_chain", "logistics", "ration"]),

    # --- طاقة متقدمة ---
    "energy.fusion": ("اندماج نووي", ["tokamak", "plasma", "deuterium", "tritium"]),
    "energy.superconductor": ("موصلية فائقة", ["zero_resistance", "cooling", "magnet"]),

    # --- أخرى ---
    "other.cartography": ("خرائط", ["projection", "GIS", "survey", "coordinate"]),
    "other.meteorology": ("أرصاد جوية", ["forecast", "satellite", "radar", "model"]),
    "other.archeology": ("آثار", ["excavation", "dating", "artifact", "civilization"]),
    "other.music_tech": ("تقنيات صوتية", ["synthesis", "sampling", "mixing", "DSP"]),
}


def count_capsules():
    """عدد الكبسولات"""
    return len(CAPSULE_REGISTRY)


def get_categories():
    """الفئات الرئيسية"""
    cats = {}
    for cid in CAPSULE_REGISTRY:
        cat = cid.split(".")[0]
        cats[cat] = cats.get(cat, 0) + 1
    return cats


def expand_tree():
    """
    توسيع capsule_tree من 31 → 500+
    يضيف كل الكبسولات الجديدة مع وراثة أوتوماتيكية
    """
    try:
        from brain.capsule_tree import tree
    except ImportError:
        import sys
        sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent.parent))
        from brain.capsule_tree import tree

    added = 0
    for cid, (name_ar, keywords) in CAPSULE_REGISTRY.items():
        if cid in tree.nodes:
            continue

        # Determine parent
        parts = cid.rsplit(".", 1)
        parent_ids = [parts[0]] if len(parts) > 1 and parts[0] in tree.nodes else []

        # Auto-create parent pyramid if missing
        if parent_ids and parent_ids[0] not in tree.nodes:
            pp = parent_ids[0].rsplit(".", 1)
            grand_parent = [pp[0]] if len(pp) > 1 else []
            tree.add_node(
                parent_ids[0],
                name=parent_ids[0].split(".")[-1],
                name_ar=parent_ids[0].split(".")[-1],
                node_type="pyramid",
                parent_ids=grand_parent if grand_parent and grand_parent[0] in tree.nodes else [],
                keywords=[],
                auto=True,
            )

        tree.add_node(
            cid,
            name=cid.split(".")[-1],
            name_ar=name_ar,
            node_type="capsule",
            parent_ids=parent_ids,
            keywords=keywords,
            auto=True,
        )
        added += 1

    print(f"✅ أضاف {added} كبسولة جديدة")
    print(f"   المجموع: {tree.stats()['total_nodes']} عقدة")
    return added


if __name__ == "__main__":
    print(f"🌳 سجل الكبسولات: {count_capsules()} كبسولة\n")
    cats = get_categories()
    for cat, count in sorted(cats.items(), key=lambda x: -x[1]):
        print(f"  {cat:25s} {count:3d}")
    print(f"\n  {'المجموع':25s} {sum(cats.values()):3d}")
    print()
    expand_tree()

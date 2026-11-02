"""Syllabus topics (CIE 9709 2023-27, 9702 2022-27) and keyword evidence for each.

Each pattern is (regex, weight).  Scores are summed over question text, and
(at lower weight) over the question's mark-scheme text.
"""
import re

# ---------------------------------------------------------------- maths 9709
MATH = {
 1: [
  ("1.1", "Quadratics", [
     (r"complet(e|ing) the square", 4), (r"in the form \(?a ?\(x ?[+\-–] ?b\)", 3), (r"in the form .{0,12}\(x ?[+\-–] ?[a-z]\) ?2", 2),
     (r"discriminant", 4), (r"b ?2 ?[-–−] ?4 ?a ?c", 4), (r"real roots|distinct roots|equal roots|no real", 3),
     (r"set of values of (the constant )?[kmpa]\b", 3), (r"values of (the constant )?[kmpa] for which", 2),
     (r"quadratic", 2), (r"meets? the curve|intersects? the curve|does not meet", 1.5), (r"is a tangent to the curve", 1.5),
     (r"vertex|minimum value of|maximum value of", 1), (r"\bk\b", 0.4)]),
  ("1.2", "Functions", [
     (r"\bf ?g ?\(|\bg ?f ?\(|\bff ?\(|\bgg ?\(|composite", 3), (r"inverse|f ?[-–−] ?1|g ?[-–−] ?1", 3), (r"\bdomain\b", 3),
     (r"\brange\b", 2.5), (r"one-one|one to one|one‑one", 3), (r"transformation|translation|stretch|reflection", 3),
     (r"graph of y ?= ?f ?\(", 2), (r"\bf ?\( ?x ?\) ?=", 1), (r"\bg ?\( ?x ?\) ?=", 1.5), (r"self-inverse|is a function", 2),
     (r"mapping|for x ?[⩾≥>]", 0.5)]),
  ("1.3", "Coordinate geometry", [
     (r"\bcircle\b", 3), (r"\bcentre\b", 2.5), (r"radius", 1.5), (r"perpendicular bisector", 4), (r"mid-?point", 2),
     (r"equation of the line", 1.5), (r"gradient of the line", 1.5), (r"perpendicular to", 1.5), (r"\bparallel\b", 1),
     (r"points? [a-z] ?\( ?[-–−]?\d", 1.5), (r"coordinates of", 1), (r"tangent to the circle|touches the", 3),
     (r"straight line|line l|line l1|lines? ?[a-z]? ?(?:with|has) equation", 1)]),
  ("1.4", "Circular measure", [
     (r"radians?\b", 3), (r"\bsector\b", 4), (r"\barc\b", 3.5), (r"perimeter", 2.5), (r"shaded region", 2),
     (r"segment", 1), (r"\brad\b", 1.5), (r"angle [a-z]{3} ?= ?.{0,6}(radians|r\b|π)", 2)]),
  ("1.5", "Trigonometry", [
     (r"\bsin\b|\bcos\b|\btan\b", 1.2), (r"θ|\bi\b", 0.6), (r"identity|≡", 2.5), (r"0 ?°|360 ?°|180 ?°|°", 1.5),
     (r"trigonometric", 2), (r"solve the equation", 1), (r"\bsin ?2|\bcos ?2|\btan ?2", 1), (r"exact value", 0.5),
     (r"y ?= ?a ?\+ ?b ?(sin|cos|tan)|y ?= ?\w? ?(sin|cos|tan)", 2)]),
  ("1.6", "Series", [
     (r"binomial", 4), (r"expansion", 3), (r"coefficient", 2.5), (r"term independent of|independent of x", 3),
     (r"arithmetic progression|\barithmetic\b", 4), (r"geometric progression|\bgeometric\b", 4),
     (r"common difference|common ratio", 4), (r"sum to infinity", 4), (r"first term|nth term|\bn ?th\b", 2),
     (r"sum of the first", 3), (r"progression|sequence", 2), (r"ascending powers", 3)]),
  ("1.7", "Differentiation", [
     (r"d ?y ?/? ?d ?x|dy|d ?x\b", 1.5), (r"stationary", 3), (r"maximum|minimum", 1.5), (r"gradient of the curve|gradient at", 2),
     (r"\btangent\b", 1.5), (r"\bnormal\b", 2), (r"rate of (change|increase|decrease)|\bincreasing\b at a rate|\brate\b", 2.5),
     (r"increasing|decreasing", 2), (r"differentiat|derivative|f ?′|f ?'", 2.5), (r"second derivative|d ?2 ?y", 2.5),
     (r"nature of", 2)]),
  ("1.8", "Integration", [
     (r"integrat|[∫ÔÓ]", 3), (r"\barea\b", 2.5), (r"\bvolume\b", 3.5), (r"rotated through|360 ?° about", 3.5),
     (r"region", 1.5), (r"find the equation of the curve", 3), (r"exact (value|area)", 1), (r"bounded by", 2),
     (r"shaded", 0.5), (r"passes through the point", 1)]),
 ],
 2: [
  ("2.1", "Algebra", [
     (r"modulus|\|", 2.5), (r"sketch,? on the same diagram,? the graphs of", 3.5), (r"\binequalit", 1.5), (r"polynomial", 3), (r"factor of|is a factor|factor theorem|factorise", 2.5),
     (r"remainder", 3.5), (r"divided by|division", 2.5), (r"quotient", 1.5), (r"inequality|satisfying the inequality", 2),
     (r"set of values of x|values of x satisfying", 2), (r"\bp ?\( ?x ?\)|\bf ?\( ?x ?\) ?= ?\d? ?x ?3", 1)]),
  ("2.2", "Logarithmic and exponential functions", [
     (r"\bln\b", 2), (r"\blog", 2.5), (r"\be ?\d? ?x|\be ?[-–−] ?x|exponential", 1.5), (r"ln ?y|lny", 3),
     (r"against x|against ln", 3), (r"straight line graph|straight line", 1), (r"\d ?x ?[+\-–] ?\d", 0.2),
     (r"logarithm", 3), (r"in the form y ?= ?a", 1)]),
  ("2.3", "Trigonometry", [
     (r"\bsec\b|\bcosec\b|\bcot\b", 2.5), (r"identity|≡", 2.5), (r"r ?cos|r ?sin|in the form r", 3.5),
     (r"\bsin\b|\bcos\b|\btan\b", 1), (r"°", 1.2), (r"double angle|sin ?2 ?[θix]|cos ?2 ?[θix]|tan ?2 ?[θix]", 1.5),
     (r"solve the equation", 0.8), (r"θ", 0.5)]),
  ("2.4", "Differentiation", [
     (r"a curve has equation|the equation of a curve is|exact gradient|exact coordinates of the stationary", 2),
     (r"d ?y ?/? ?d ?x|\bdy\b", 1.5), (r"stationary", 3), (r"parametric", 4), (r"implicit", 3), (r"tangent|normal", 1.5),
     (r"gradient", 1.5), (r"differentiat|derivative", 2.5), (r"maximum|minimum", 1), (r"quotient rule|product rule", 3)]),
  ("2.5", "Integration", [
     (r"integrat|[∫ÔÓ]", 3), (r"exact value", 1.2), (r"\barea\b", 2), (r"region", 1.5), (r"trapezium rule", 4),
     (r"bounded by", 1.5), (r"d ?x\b", 0.3)]),
  ("2.6", "Numerical solution of equations", [
     (r"iterat", 4), (r"x ?n ?\+ ?1", 3), (r"converge", 3), (r"root lies|lies between", 3), (r"sketching a suitable pair", 4),
     (r"exactly one root|only one root|real root", 2), (r"decimal places", 1), (r"sign change|change of sign", 3)]),
 ],
 3: [
  ("3.1", "Algebra", [
     (r"modulus|\|", 2.5), (r"sketch,? on the same diagram,? the graphs of", 3.5), (r"\binequalit", 1.5), (r"polynomial", 3), (r"factor of|is a factor|factor theorem|factorise", 2.5),
     (r"remainder", 3.5), (r"divided by|division", 2), (r"partial fractions", 4), (r"ascending powers", 3),
     (r"binomial|expansion", 2.5), (r"inequality|satisfying the inequality", 2), (r"set of values of x", 1.5),
     (r"coefficient of", 1.5)]),
  ("3.2", "Logarithmic and exponential functions", [
     (r"\bln\b", 1.5), (r"\blog", 2.5), (r"exponential", 1), (r"ln ?y|lny", 3), (r"against x|against ln", 3),
     (r"logarithm", 3), (r"straight line", 1), (r"solve the equation .{0,40}(\be|\d ?x)", 0.8)]),
  ("3.3", "Trigonometry", [
     (r"\bsec\b|\bcosec\b|\bcot\b", 2.5), (r"identity|≡", 2.5), (r"r ?cos|r ?sin|in the form r", 3.5),
     (r"\bsin\b|\bcos\b|\btan\b", 0.8), (r"°", 1.2), (r"double angle|sin ?2 ?[θix]|cos ?2 ?[θix]|tan ?2 ?[θix]", 1.2),
     (r"solve the equation", 0.6), (r"θ", 0.5), (r"prove", 0.8)]),
  ("3.4", "Differentiation", [
     (r"a curve has equation|the equation of a curve is|exact gradient|exact coordinates of the stationary", 2),
     (r"d ?y ?/? ?d ?x|\bdy\b", 1.5), (r"stationary", 3), (r"parametric", 4), (r"implicit", 3), (r"tangent|normal", 1.5),
     (r"gradient", 1.5), (r"differentiat|derivative", 2.5), (r"maximum|minimum", 1.2), (r"quotient rule|product rule|chain rule", 3)]),
  ("3.5", "Integration", [
     (r"integrat|[∫ÔÓ]", 3), (r"exact value", 1.5), (r"\barea\b", 2), (r"region", 1.5), (r"substitution", 3), (r"by parts", 4),
     (r"bounded by", 1.5), (r"d ?x\b", 0.3), (r"\bdu\b|d ?u\b", 1)]),
  ("3.6", "Numerical solution of equations", [
     (r"iterat", 4), (r"x ?n ?\+ ?1", 3), (r"converge", 3), (r"root lies|lies between", 3), (r"sketching a suitable pair", 4),
     (r"exactly one root|only one root|real root", 2), (r"decimal places", 0.8), (r"sign change|change of sign", 3)]),
  ("3.7", "Vectors", [
     (r"vector", 3.5), (r"position vectors?", 3), (r"\bi ?[+\-–] ?\d? ?j|\bj ?[+\-–] ?\d? ?k", 3), (r"\blines? l", 2),
     (r"r ?= ?", 1), (r"skew", 4), (r"perpendicular", 1), (r"\bo[abcdp]\b", 1.5), (r"foot of the perpendicular", 3),
     (r"parallelogram|trapezium|cuboid|pyramid", 1.5), (r"angle between", 1.5)]),
  ("3.8", "Differential equations", [
     (r"differential equation", 8), (r"variables? \w+ and \w+ satisfy", 2), (r"d ?y ?/? ?d ?t|d ?x ?/? ?d ?t|d ?\w ?/ ?d ?t|\bdt\b|d ?t\b", 1.5),
     (r"proportional to", 2.5), (r"rate of", 1.5), (r"separat", 3), (r"general solution|particular solution", 2)]),
  ("3.9", "Complex numbers", [
     (r"complex", 4), (r"argand", 4), (r"\bi\b", 0.8), (r"\barg\b|argument", 3), (r"modulus", 1), (r"conjugate|z ?\*", 3),
     (r"\bz\b", 1.5), (r"\bu ?=|\bw ?=", 0.8), (r"\bre\b|\bim\b|real part|imaginary", 2), (r"square roots? of", 1.5)]),
 ],
 4: [
  ("4.1", "Forces and equilibrium", [
     (r"equilibrium", 4), (r"held (at rest|in equilibrium)|remains? at rest|is at rest", 3), (r"limiting|about to (slip|slide|move)", 3), (r"coplanar|forces of magnitudes?", 3),
     (r"resultant", 2), (r"components?", 1.2), (r"at rest on", 1.2), (r"smallest .{0,20}force|least .{0,20}force", 2),
     (r"coefficient of friction", 1), (r"in equilibrium", 2), (r"resolv", 1)]),
  ("4.2", "Kinematics of motion in a straight line", [
     (r"velocity-time|velocity–time|v-t|displacement-time|displacement–time|graph", 2.5), (r"displacement", 2.5),
     (r"moves in a straight line", 2.5), (r"\bt ?s after|\bt ?seconds? after|at time t", 2), (r"instantaneous(ly)? at rest|comes to rest", 1.5),
     (r"distance travelled", 2), (r"acceleration of .{0,10}is given|v ?= ?\d?\.?\d* ?t|a ?= ?\d?\.?\d* ?t", 2.5),
     (r"maximum velocity|maximum speed", 1), (r"decelerat|constant acceleration", 1), (r"\bsuvat|\bv ?2", 0.3)]),
  ("4.3", "Momentum", [
     (r"momentum", 4), (r"collid|collision", 3.5), (r"coalesce", 4), (r"rebound|reverses? direction", 2), (r"spheres?", 1.5),
     (r"impulse", 3), (r"moving towards each other|in opposite directions", 1.5)]),
  ("4.4", "Newton's laws of motion", [
     (r"pulley", 3), (r"light inextensible string", 2), (r"connected|attached to", 1), (r"tow-?bar|trailer|caravan", 1.5),
     (r"tension", 1.5), (r"driving force", 1), (r"\blift\b", 2.5), (r"released from rest", 1.5), (r"accelerat", 1),
     (r"rough (plane|horizontal|surface|inclined)|smooth (plane|horizontal|surface|inclined)", 1),
     (r"newton's second law|n2l", 2), (r"resistance", 0.5)]),
  ("4.5", "Energy, work and power", [
     (r"work done", 3.5), (r"\bpower\b|\bkw\b|\bw\b", 2.5), (r"kinetic energy|potential energy|\benergy\b", 3),
     (r"work-energy|work–energy", 3), (r"working at|engine", 1.5), (r"maximum speed|maximum constant speed|steady speed", 1)]),
 ],
 5: [
  ("5.1", "Representation of data", [
     (r"stem-and-leaf|stem and leaf", 5), (r"box-and-whisker|box and whisker|box plot", 5), (r"cumulative frequency", 4),
     (r"histogram", 5), (r"frequency density", 4), (r"interquartile|quartile", 3.5), (r"\bmedian\b", 1.5),
     (r"standard deviation", 1), (r"σ ?x|σx|sum of|Σ", 1.2), (r"outlier", 3), (r"class width|grouped", 2.5),
     (r"frequency", 1), (r"\bmean\b", 0.6), (r"\bdata\b", 1.2), (r"variance", 0.5)]),
  ("5.2", "Permutations and combinations", [
     (r"arrangements?|arranged", 4), (r"letters", 3), (r"how many (different )?ways|number of (different )?ways|\bways\b", 3.5),
     (r"selections?|selected|chosen|choose", 2.5), (r"committee|team|group of \d", 2), (r"next to each other|together|in a row|in a line", 2.5),
     (r"how many different", 2.5), (r"positions", 1), (r"digits", 1.5)]),
  ("5.3", "Probability", [
     (r"tree diagram", 4), (r"independent", 2), (r"mutually exclusive|exclusive", 3), (r"given that", 1.5),
     (r"conditional", 3), (r"\bevents?\b|\bp ?\( ?[a-z] ?(∩|∪|\|)", 2.5), (r"dice|die\b|spinner|coin", 1.5),
     (r"probability", 1), (r"\bbag\b|balls|counters", 1), (r"venn", 3)]),
  ("5.4", "Discrete random variables", [
     (r"probability distribution", 4), (r"random variable", 2.5), (r"e ?\( ?x ?\)|var ?\( ?x ?\)|expectation|expected", 2),
     (r"binomial|b ?\( ?\d+ ?,|geometric distribution|geo ?\(", 3.5), (r"\btable\b", 0.8), (r"\bx\b", 0.2),
     (r"number of (sixes|successes|times|heads)", 1.5), (r"first (time|occasion|success)|before the first", 4),
     (r"the random variable [a-z]", 2.5)]),
  ("5.5", "The normal distribution", [
     (r"normally distributed|normal distribution", 5), (r"\bn ?\( ?\d", 2.5), (r"standard deviation", 1.5),
     (r"approximation", 2), (r"\bz\b|φ|Φ", 1.5), (r"continuity correction", 3), (r"\bmean\b", 0.5)]),
 ],
 6: [
  ("6.1", "The Poisson distribution", [
     (r"poisson|\bpo ?\(", 4), (r"per (hour|minute|day|week|month|year|second|page|metre|km|square)", 2), (r"at random|randomly", 1),
     (r"occur", 1), (r"approximat", 1), (r"average (rate|number)|on average", 1.5)]),
  ("6.2", "Linear combinations of random variables", [
     (r"independent random variables|independent", 1.5), (r"\b(total|sum) (mass|weight|time|length)|total of", 2.5),
     (r"[xy] ?1 ?\+ ?[xy] ?2|\d ?[xy] ?[-–−+] ?\d? ?[xy]\b", 2.5), (r"var ?\( ?\d? ?[xy] ?[+\-–−]", 3),
     (r"e ?\( ?\d ?x ?[+\-–−]|linear combination", 3), (r"difference between", 1.5), (r"exceeds|more than twice|greater than", 1),
     (r"randomly chosen .{0,40}(and|plus) .{0,40}randomly chosen", 1)]),
  ("6.3", "Continuous random variables", [
     (r"probability density function|density function|pdf", 5), (r"f ?\( ?x ?\) ?= ?\{|f ?\( ?x ?\) ?= ?[a-z0-9]", 1),
     (r"otherwise", 2.5), (r"\bmedian\b", 1), (r"\bk\b", 0.5), (r"integrat|∫", 1), (r"for \d ?[⩽≤<] ?x ?[⩽≤<]", 1.5)]),
  ("6.4", "Sampling and estimation", [
     (r"unbiased estimate", 4), (r"confidence interval", 5), (r"sample mean|distribution of the sample mean|central limit", 3.5),
     (r"random sample", 1.5), (r"sampling|sample of", 1.2), (r"\bx ?̄|x‾|x̄", 1), (r"estimate", 1), (r"simple random sample|random numbers", 2.5)]),
  ("6.5", "Hypothesis tests", [
     (r"hypothes", 3.5), (r"significance level|% (level of )?significance|\d+ ?% (significance )?level", 4),
     (r"type i\b|type ii|type 1|type 2", 4), (r"critical region|rejection region", 4), (r"\btest\b|tested|testing", 1.5),
     (r"\bh ?0\b|\bh ?1\b", 3), (r"claim|suspects?|believes?", 1.2)]),
 ],
}

# --------------------------------------------------------------- physics 9702
PHYS = [
 ("01", "Physical quantities and units", [
   (r"base units?|si units?|homogeneous|in terms of (the )?(si )?base", 4), (r"uncertaint", 3.5), (r"systematic|random error|zero error", 3),
   (r"accuracy|precision|precise|accurate", 2.5), (r"\bvectors?\b|\bscalars?\b|resolv|components? of", 2), (r"estimate|order of magnitude|reasonable", 2),
   (r"prefix|\bpico|\bnano|\bmicro|\bmega|\bgiga|\btera", 1.5), (r"micrometer|vernier|calipers|metre rule", 2), (r"significant figures", 1),
   (r"physical quantit|derived unit", 3)]),
 ("02", "Kinematics", [
   (r"displacement", 2), (r"velocity", 1.5), (r"acceleration", 1.2), (r"projectile|thrown|projected|launched|kicked|fired", 3),
   (r"free fall|freely|falls? from rest|dropped", 2), (r"horizontal(ly)? (component|distance|velocity)|vertical(ly)? (component|velocity)", 2.5),
   (r"velocity[-–]time|speed[-–]time|displacement[-–]time|distance[-–]time", 3), (r"uniform(ly)? accelerat|constant acceleration", 2),
   (r"air resistance is negligible", 1.5), (r"equations? of motion", 2)]),
 ("03", "Dynamics", [
   (r"momentum", 3.5), (r"collid|collision|elastic|inelastic|rebound", 3), (r"newton'?s (first|second|third) law|third law", 3.5),
   (r"resultant force", 1.5), (r"terminal (velocity|speed)|drag|viscous|air resistance", 2.5), (r"impulse", 2.5),
   (r"\bmass\b", 0.4), (r"weight", 0.5), (r"inertia", 2), (r"force[-–]time", 2)]),
 ("04", "Forces, density and pressure", [
   (r"moment|torque|couple|pivot|turning effect|centre of gravity|centre of mass", 3), (r"equilibrium", 2),
   (r"density", 2.5), (r"pressure", 2.5), (r"upthrust|archimedes|immersed|submerged|floats?\b|buoyan", 3.5),
   (r"fluid|liquid|water", 0.6), (r"tension", 0.6), (r"frictional?", 0.6), (r"\bρ ?g ?h|depth", 1.5)]),
 ("05", "Work, energy and power", [
   (r"work done|\bwork\b", 2), (r"kinetic energy", 2.5), (r"potential energy", 2), (r"\bpower\b", 2), (r"efficien", 3),
   (r"conservation of energy|energy (is )?(transferred|converted|dissipated)|useful", 2), (r"\benergy\b", 0.8), (r"\bwatts?\b|kw\b", 1.5)]),
 ("06", "Deformation of solids", [
   (r"young modulus", 4.5), (r"stress|strain", 3.5), (r"extension|stretch(ed)?|compress", 2), (r"spring constant|springs?\b", 2.5),
   (r"hooke|elastic limit|limit of proportionality|plastic|elastic deformation", 3.5), (r"strain energy|elastic potential energy", 3.5),
   (r"wire", 1), (r"load", 1.5), (r"force[-–]extension|stress[-–]strain", 3)]),
 ("07", "Waves", [
   (r"wavelength", 1.5), (r"frequency", 0.8), (r"amplitude", 1), (r"transverse|longitudinal", 3), (r"doppler", 4),
   (r"electromagnetic (wave|spectrum|radiation)|spectrum|ultraviolet|infrared|microwave|x-ray|radio wave", 2.5),
   (r"polari", 3.5), (r"malus", 4), (r"intensity", 1.5), (r"progressive", 3), (r"phase difference", 2), (r"cathode-ray oscilloscope|oscilloscope|time-base|y-gain", 2.5),
   (r"speed of sound|sound wave|loudspeaker|microphone", 1.2), (r"rarefaction|compression", 2)]),
 ("08", "Superposition", [
   (r"stationary wave|standing wave", 4), (r"\bnodes?\b|antinodes?", 3.5), (r"interference|fringe|coherent|coherence|path difference", 3.5),
   (r"diffraction|grating|double[- ]slit|two[- ]source|young", 3.5), (r"superposition|superpose", 3), (r"harmonic|resonan|air column|tube", 1.5),
   (r"slits?\b", 2), (r"maxima|minima|maximum intensity|order", 1)]),
 ("09", "Electricity", [
   (r"\bcurrent\b", 1), (r"\bcharge\b|coulomb", 1.2), (r"resistivity", 4), (r"resistance", 1), (r"i[-–]v characteristic|current[-–]voltage|characteristic", 3),
   (r"thermistor|light-dependent|ldr|filament|diode|lamp", 2), (r"number density|drift|electrons? per", 2.5), (r"cross-sectional area", 1.5),
   (r"potential difference", 0.8), (r"power (dissipated|in the)|kilowatt", 1), (r"ohm'?s law", 2)]),
 ("10", "D.C. circuits", [
   (r"kirchhoff", 4), (r"series|parallel", 1.5), (r"potential divider|potentiometer", 4), (r"internal resistance", 4), (r"e\.m\.f\.|electromotive", 2.5),
   (r"terminal (potential difference|p\.d\.)|lost volts", 3), (r"circuit", 1), (r"\bcell\b|battery|supply", 1), (r"galvanometer|null", 2.5),
   (r"combined resistance|total resistance|equivalent resistance", 2.5), (r"voltmeter|ammeter", 1)]),
 ("11", "Particle physics", [
   (r"quark|lepton|hadron|baryon|meson", 4.5), (r"neutrino|positron|antiparticle|antineutrino|electron neutrino", 3.5), (r"β|beta|alpha|α", 2),
   (r"nucleus|nuclei|nucleon|nuclide|proton number|nucleon number|isotope", 2.5), (r"fundamental particle|strange|up quark|down quark", 3),
   (r"scattering|gold foil", 3), (r"decay", 1.5), (r"proton|neutron", 1.2)]),
 ("12", "Motion in a circle", [
   (r"angular (speed|velocity|displacement)", 3.5), (r"centripetal", 4), (r"circular (path|motion|orbit)|in a circle|horizontal circle|vertical circle", 3),
   (r"radians?\b|rad s", 1.5), (r"revolution|rotat", 1.5), (r"period of rotation|rev(olutions)? per", 1.5)]),
 ("13", "Gravitational fields", [
   (r"gravitational (field|potential|force)", 3.5), (r"orbit|satellite|geostationary|planet|moon|sun\b|star\b", 2.5),
   (r"newton'?s law of gravitation|universal gravitational|gravitational constant", 4), (r"point mass|escape", 2.5), (r"earth", 1)]),
 ("14", "Temperature", [
   (r"thermodynamic temperature|kelvin|absolute zero|thermometer|thermal equilibrium", 3.5), (r"specific heat capacity", 4),
   (r"specific latent heat|latent heat", 4), (r"melting|boiling|vaporis|evaporat|fusion of ice|ice", 1.5), (r"°c|celsius", 1.5), (r"heater", 1)]),
 ("15", "Ideal gases", [
   (r"ideal gas", 4), (r"\bmoles?\b|avogadro|boltzmann|molar", 3), (r"kinetic theory|root-mean-square|r\.m\.s\. speed|mean square speed", 4),
   (r"\bmolecules?\b|atoms of the gas", 1.5), (r"\bgas\b", 1.5), (r"pv ?= ?nrt|pv ?= ?nkt", 3), (r"brownian", 3)]),
 ("16", "Thermodynamics", [
   (r"internal energy", 4), (r"first law of thermodynamics", 4.5), (r"work done on the gas|work done by the gas|heating|thermal energy (transferred|supplied)", 3),
   (r"compress|expan(ds|sion)", 1.5), (r"cycle|p[-–]v|pressure[-–]volume", 2)]),
 ("17", "Oscillations", [
   (r"simple harmonic|s\.h\.m\.|\bshm\b", 4.5), (r"oscillat", 2.5), (r"damp", 3.5), (r"resonan|natural frequency|forced", 3),
   (r"angular frequency", 2.5), (r"pendulum|spring|mass-spring|trolley", 1.2), (r"period\b", 0.8), (r"acceleration .{0,30}proportional", 2)]),
 ("18", "Electric fields", [
   (r"electric field", 3.5), (r"electric potential|potential gradient|equipotential", 3), (r"coulomb'?s law|point charges?|charged spheres?", 3.5),
   (r"parallel plates?|between the plates|metal plates", 2), (r"field lines", 1.5), (r"ε ?0|permittivity", 2), (r"isolated (metal )?sphere", 2)]),
 ("19", "Capacitance", [
   (r"capacit", 4.5), (r"time constant", 3.5), (r"discharg", 3), (r"farad|μf|\bpf\b", 2.5), (r"charge stored|energy stored", 1.5)]),
 ("20", "Magnetic fields", [
   (r"magnetic (field|flux)", 3.5), (r"flux (density|linkage)|\btesla\b", 3), (r"hall", 4), (r"fleming|left-hand rule", 3),
   (r"faraday|lenz|induc(ed|tion)", 3.5), (r"solenoid|coil", 2), (r"current-carrying|wire carrying", 2), (r"velocity selector|cyclotron", 3),
   (r"magnet", 1.5)]),
 ("21", "Alternating currents", [
   (r"alternating", 3.5), (r"r\.m\.s\.|root-mean-square|\brms\b", 2.5), (r"peak (value|voltage|current)", 2.5), (r"rectif", 4),
   (r"smoothing|smoothed|ripple", 3.5), (r"sinusoidal", 2), (r"mean power", 2.5), (r"diodes?\b", 1.5), (r"transformer", 2)]),
 ("22", "Quantum physics", [
   (r"photons?", 3), (r"photoelectric", 4.5), (r"work function|threshold frequency", 4), (r"de broglie", 4.5), (r"energy levels?", 3.5),
   (r"line spectr|emission spectr|absorption spectr|spectral lines?", 3.5), (r"planck", 2.5), (r"wave-particle|electron diffraction", 3),
   (r"electronvolt|\bev\b", 1.2)]),
 ("23", "Nuclear physics", [
   (r"mass defect|binding energy", 4.5), (r"fission|fusion", 3.5), (r"half-life|half life", 4), (r"decay constant|activity|becquerel", 3.5),
   (r"radioactive|radioisotope|radionuclide", 2.5), (r"spontaneous|random nature", 2), (r"count rate|background", 2), (r"nucleus|nuclei|nucleon", 1),
   (r"\bu\b|unified atomic mass", 1)]),
 ("24", "Medical physics", [
   (r"ultrasound|ultrasonic|piezo", 4.5), (r"acoustic impedance|specific acoustic", 4.5), (r"x-ray|x-rays|attenuation|linear attenuation", 3.5),
   (r"\bct\b|computed tomography|tomograph|voxel", 4), (r"\bpet\b|positron emission|tracer|annihilat|gamma camera|γ-ray photons", 4),
   (r"contrast|sharpness|image", 1.5), (r"patient|body|tissue|bone|medical|diagnos", 2)]),
 ("25", "Astronomy and cosmology", [
   (r"luminosity", 4.5), (r"radiant flux intensity", 4.5), (r"standard candle", 4.5), (r"stefan|wien", 4), (r"redshift|red-shift|red shift", 4),
   (r"hubble", 4.5), (r"galax|universe|big bang|cosmolog", 3.5), (r"\bstars?\b|stellar|sun\b", 1.5), (r"light-years?|parsec", 2)]),
]

# Mark-scheme method cues (applied to mark-scheme text at full weight)
MS_METHOD = {
 "1.1": r"discriminant|b ?2 ?- ?4 ?ac|complet\w* (the )?square|critical values?|quadratic formula",
 "1.2": r"\brange\b|\bdomain\b|inverse|composite|\bf ?g ?\(|\bg ?f ?\(|translation|stretch|reflection|one-one",
 "1.3": r"gradient of (the )?(line|ab|normal)|perpendicular|mid-?point|equation of (the )?(line|circle)|centre|radius|distance between",
 "1.4": r"arc length|sector|perimeter|radians|segment|r ?θ|1 ?/? ?2 ?r ?2",
 "1.5": r"identity|sin ?2 ?θ ?\+ ?cos ?2|tan ?θ ?=|θ ?=|cos ?θ ?=|sin ?θ ?=",
 "1.6": r"binomial|coefficient|\bap\b|\bgp\b|common (ratio|difference)|sum to infinity|\bs ?n ?=|\bn ?c ?r|\d ?c ?\d",
 "1.7": r"differentiat|d ?y ?/ ?d ?x|stationary|second derivative|rate of change|chain rule|d ?2 ?y",
 "1.8": r"integrat|\blimits\b|\barea\b|\bvolume\b|π ?∫|\+ ?c\b",
 "2.1": r"remainder|quotient|modulus|critical values?|factor theorem|long division|division|\bsquar(e|ing) both sides",
 "2.2": r"\blog|logarithm|take logs|ln ?y ?=|gradient of (the )?(line|graph)|intercept",
 "2.3": r"identity|\br ?=|tan ?α|\bα ?=|\bsec\b|\bcosec\b|\bcot\b|double angle|cos ?2 ?θ|sin ?2 ?θ",
 "2.4": r"differentiat|product rule|quotient rule|chain rule|implicit|parametric|stationary|d ?y ?/ ?d ?x",
 "2.5": r"integrat|integrand|\blimits\b|trapezium|\barea\b",
 "2.6": r"iterat|converge|sign change|change of sign|\bx ?n ?\+ ?1",
 "3.1": r"remainder|quotient|partial fraction|modulus|critical values?|factor theorem|binomial|expansion|\bsquar(e|ing) both sides",
 "3.2": r"\blog|logarithm|take logs|ln ?y ?=|gradient of (the )?(line|graph)|intercept",
 "3.3": r"identity|\br ?=|tan ?α|\bα ?=|\bsec\b|\bcosec\b|\bcot\b|double angle|cos ?2 ?θ|sin ?2 ?θ",
 "3.4": r"differentiat|product rule|quotient rule|chain rule|implicit|parametric|stationary|d ?y ?/ ?d ?x",
 "3.5": r"integrat|integrand|\blimits\b|substitut|by parts|\barea\b",
 "3.6": r"iterat|converge|sign change|change of sign|\bx ?n ?\+ ?1",
 "3.7": r"vector|direction|scalar product|position vector|\bλ|\bμ|parameter",
 "3.8": r"separat\w* (the )?variables|separation of variables|differential ?equation|integrate both sides|arbitrary constant",
 "3.9": r"complex|argand|\barg\b|argument|conjugate|imaginary|half-line|locus|circle|i ?2 ?= ?-1",
 "4.1": r"resolv|equilibrium|limiting|f ?= ?μ ?r|lami|triangle of forces",
 "4.2": r"suvat|v ?= ?u|s ?= ?ut|area under|displacement|integrat|differentiat|\bv ?=|\bs ?=",
 "4.3": r"momentum|colm|coalesce|before and after",
 "4.4": r"newton'?s second law|\bn2l\b|f ?= ?ma|connected particles|whole system|tension",
 "4.5": r"work done|\bwd\b|\bpe\b|\bke\b|work-energy|work–energy|\bpower\b|p ?= ?f ?v|energy",
 "5.1": r"median|quartile|\biqr\b|frequency density|histogram|class width|cumulative|standard deviation|σ ?x|outlier",
 "5.2": r"arrangement|\bways\b|\d ?c ?\d|\d ?p ?\d|factorial|\d ?!|scenarios?",
 "5.3": r"tree|conditional|independent|exclusive|\bp ?\( ?[a-z] ?(∩|∪|\|)|given",
 "5.4": r"probability distribution|e ?\( ?x ?\)|var ?\( ?x ?\)|binomial|geometric|table|σ ?p ?x",
 "5.5": r"\bz ?=|φ|standardis|normal|continuity correction|\bn ?\(",
 "6.1": r"poisson|\bpo ?\(|\bλ|e ?- ?λ|e ?- ?\d",
 "6.2": r"var ?\(|e ?\(|independent|linear combination|sum of|difference",
 "6.3": r"integrat|[∫ÔÓ]|median|pdf|\bk ?=|area under",
 "6.4": r"unbiased|estimate|confidence interval|sample mean|central limit|clt|n ?- ?1",
 "6.5": r"\bh ?0|\bh ?1|significance|critical (region|value)|type i|type ii|reject|\btest\b",
}

PHYS_P5 = [("P1", "Planning (Paper 5 Q1)"), ("P2", "Data analysis and evaluation (Paper 5 Q2)")]

_compiled = {}
def compiled(spec):
    key = id(spec)
    if key not in _compiled:
        # text extraction sometimes drops spaces ("solvethedifferentialequation"): make literal spaces optional
        _compiled[key] = [(code, name, [(re.compile(re.sub(r"(?<!\\) (?![?*+])", " ?", p), re.I), w) for p, w in pats])
                          for code, name, pats in spec]
    return _compiled[key]

import unicodedata
def norm(text):
    t = unicodedata.normalize("NFKC", text)
    t = t.replace("−", "-").replace("–", "-").replace("—", "-").replace("ﬁ", "fi").replace("ﬂ", "fl")
    t = re.sub(r"\.{4,}", " ", t)
    t = re.sub(r"\[\d+\]|\[Total:\s*\d+\]", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()

_ms_rx = {}
def scores(spec, qtext, mstext="", ms_weight=0.5, ms_method_weight=2.0):
    out = []
    q, m = norm(qtext).lower(), norm(mstext).lower()
    for code, name, pats in compiled(spec):
        s = 0.0
        for rx, w in pats:
            s += w * min(len(rx.findall(q)), 3)
            if m:
                s += ms_weight * w * min(len(rx.findall(m)), 3)
        if m and code in MS_METHOD:
            rx = _ms_rx.setdefault(code, re.compile(MS_METHOD[code], re.I))
            s += ms_method_weight * min(len(rx.findall(m)), 4)
        out.append(s)
    return out

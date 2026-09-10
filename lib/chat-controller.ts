import { explainAgronomicFit } from "./agronomy/crop-rules";
import { runLlmText } from "./llm-client";
import { loadDatasetRows } from "./recommendation/dataset";

export type ChatIntent =
  | "greeting"
  | "help"
  | "crop_recommendation"
  | "seed_recommendation"
  | "soil_suitability"
  | "seed_soil_match"
  | "season_recommendation"
  | "fertilizer_guidance"
  | "disease_query"
  | "weather_related"
  | "yield_estimation"
  | "image_analysis"
  | "unknown";

export type ResponseMode = "direct" | "full_answer" | "partial_answer_with_followup" | "fallback";

export interface ChatSessionMemory {
  intent: ChatIntent | null;
  crop: string | null;
  seed: string | null;
  location: string | null;
  soil_type: string | null;
  irrigation: string | null;
  season: string | null;
  moisture: string | null;
  land_quality: string | null;
  previous_crop: string | null;
  growth_stage: string | null;
  soil_condition: string | null;
  symptoms: string | null;
  area: string | null;
  image_present: boolean;
}

interface SessionRecord {
  sessionId: string;
  memory: ChatSessionMemory;
  updatedAt: number;
}

export interface ControlledLLMResponse {
  intent: ChatIntent;
  response_mode: ResponseMode;
  title: string;
  recommendation: string;
  suitable_conditions: string[];
  why: string;
  missing_details_needed: string[];
  follow_up_question: string;
  final_answer: string;
  quick_actions: string[];
  known_fields: Partial<ChatSessionMemory>;
}

export interface HandleUserMessageParams {
  message: string;
  sessionId?: string;
  history?: Array<{ role?: string; content?: string }>;
  imagePresent?: boolean;
}

const TTL_MS = 1000 * 60 * 60 * 4;
const sessions = new Map<string, SessionRecord>();

const DEFAULT_MEMORY: ChatSessionMemory = {
  intent: null,
  crop: null,
  seed: null,
  location: null,
  soil_type: null,
  irrigation: null,
  season: null,
  moisture: null,
  land_quality: null,
  previous_crop: null,
  growth_stage: null,
  soil_condition: null,
  symptoms: null,
  area: null,
  image_present: false
};

const REQUIRED_FIELDS: Record<ChatIntent, string[]> = {
  greeting: [],
  help: [],
  crop_recommendation: ["location_or_season", "soil_or_land_quality"],
  seed_recommendation: ["crop", "location_or_season"],
  soil_suitability: ["crop", "soil_type"],
  seed_soil_match: ["crop_or_seed", "soil_type"],
  season_recommendation: ["crop_or_location"],
  fertilizer_guidance: ["crop", "growth_stage_or_soil_condition"],
  disease_query: ["crop", "symptoms"],
  weather_related: ["location"],
  yield_estimation: ["crop", "area", "soil_or_irrigation"],
  image_analysis: ["image_present"],
  unknown: []
};

const FOLLOW_UP_PRIORITY: Record<string, string[]> = {
  crop_recommendation: ["location_or_season", "soil_type", "irrigation", "moisture", "land_quality", "previous_crop"],
  seed_recommendation: ["crop", "location_or_season", "irrigation", "soil_type"],
  disease_query: ["crop", "symptoms", "image_present"],
  seed_soil_match: ["soil_type", "location", "irrigation"],
  soil_suitability: ["soil_type", "moisture", "location"],
  season_recommendation: ["crop_or_location"],
  fertilizer_guidance: ["crop", "growth_stage", "soil_condition"],
  yield_estimation: ["crop", "area", "soil_or_irrigation"]
};

const CROP_KEYWORDS = [
  "wheat",
  "rice",
  "cotton",
  "soybean",
  "soyabean",
  "maize",
  "bajra",
  "jowar",
  "tomato",
  "onion",
  "sugarcane",
  "chana",
  "tur",
  "moong",
  "groundnut",
  "mustard",
  "gram",
  "pulses"
] as const;

const SOIL_KEYWORDS = [
  "medium black soil",
  "alluvial soil",
  "sandy loam",
  "sandy soil",
  "clay loam",
  "clay soil",
  "black soil",
  "red soil",
  "loamy soil",
  "loam",
  "clayey",
  "alluvial"
] as const;

const STATES = [
  "andhra pradesh",
  "arunachal pradesh",
  "assam",
  "bihar",
  "chhattisgarh",
  "goa",
  "gujarat",
  "haryana",
  "himachal pradesh",
  "jharkhand",
  "karnataka",
  "kerala",
  "madhya pradesh",
  "maharashtra",
  "manipur",
  "meghalaya",
  "mizoram",
  "nagaland",
  "odisha",
  "punjab",
  "rajasthan",
  "sikkim",
  "tamil nadu",
  "telangana",
  "tripura",
  "uttar pradesh",
  "uttarakhand",
  "west bengal"
];

const QUICK_ACTIONS = [
  "Recommend crop",
  "Check soil suitability",
  "Best seed for my land",
  "Disease help",
  "Fertilizer guidance",
  "Yield estimate"
];

const GENERIC_FILLER_PATTERNS = [
  /can be discussed/i,
  /can be assessed/i,
  /details already shared/i,
  /based on available details/i,
  /using the details already shared/i,
  /with the details already available/i
];

const NON_LOCATION_PHRASES = [
  "irrigated land",
  "rainfed land",
  "my field",
  "the field",
  "field",
  "land",
  "loamy soil",
  "sandy loam",
  "clay loam",
  "black soil"
];

function generateSessionId() {
  return `chat_${Date.now()}_${Math.random().toString(36).slice(2, 10)}`;
}

function cleanupSessions() {
  const now = Date.now();
  for (const [key, value] of sessions.entries()) {
    if (now - value.updatedAt > TTL_MS) {
      sessions.delete(key);
    }
  }
}

function titleCase(value: string) {
  return value
    .trim()
    .split(/\s+/)
    .filter(Boolean)
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
}

export function normalizeText(text: string) {
  return text.replace(/\s+/g, " ").trim().toLowerCase();
}

export function detectIntent(text: string): ChatIntent {
  if (/^(hi|hello|hey|namaste)\b/.test(text)) return "greeting";
  if (/\b(help|what can you do|how can you help)\b/.test(text)) return "help";
  if (/\b(image|photo|picture|leaf image|upload image|analyze image)\b/.test(text)) return "image_analysis";
  if (/\b(disease|yellow leaves|spots|fungus|fungal|pest|infection|aphids|stem borer|blight|wilting)\b/.test(text)) return "disease_query";
  if (/\b(fertilizer|urea|npk|dap|potash|nutrient|manure)\b/.test(text)) return "fertilizer_guidance";
  if (/\b(weather|temperature|rainfall|humidity|forecast)\b/.test(text)) return "weather_related";
  if (/\b(yield|production|estimate|acre|acres|hectare|hectares)\b/.test(text)) return "yield_estimation";
  if (/\b(soil and seed|seed suitable|soil suitable|soil and seed suitable|best soil and seed)\b/.test(text)) return "seed_soil_match";
  if (
    /\b(seed variety|which seed|best seed|seed recommendation|suggest seed|recommend seed)\b/.test(text) ||
    (/\b(seed|variety)\b/.test(text) && /\b(recommend|suggest|best|suits?|fit|good)\b/.test(text)) ||
    /\brecommend\s+(?:a\s+)?[a-z\s]+\s+seed\b/.test(text) ||
    /\bsuggest\s+seed\s+variety\b/.test(text)
  ) {
    return "seed_recommendation";
  }
  if (
    /\b(which crop|which crops|recommend crop|recommend crops|suitable crop|suitable crops|what should i grow|best crop|best crops)\b/.test(text) ||
    (/\b(crop|crops)\b/.test(text) && /\b(best|perform best|good|suitable|recommend|grow)\b/.test(text)) ||
    (!/\bseed|seeds\b/.test(text) && /\b(soil|loam|clay|sandy|alluvial|black soil)\b/.test(text) && /\b(kharif|rabi|zaid)\b/.test(text) && /\bwhat|which\b/.test(text))
  ) {
    return "crop_recommendation";
  }
  if (
    /\b(which soil|what soil|best soil|soil required|suitable soil|is this soil good|soil is suitable|my soil is suitable|soil suitable for)\b/.test(text) ||
    (/\b(soil|loam|clay|sandy|alluvial|black soil)\b/.test(text) && /\b(suitable for|good for|best for)\b/.test(text)) ||
    (/\bcompare\b/.test(text) && /\bsoil\b/.test(text)) ||
    /\bis\s+[a-z\s]+soil\s+(?:suitable|good)\s+for\b/.test(text)
  ) {
    return "soil_suitability";
  }
  if (/\b(season|kharif|rabi|zaid|sowing time|when to sow)\b/.test(text)) return "season_recommendation";
  return "unknown";
}

function findFirstKeyword(text: string, words: readonly string[]) {
  return [...words]
    .filter((word) => text.includes(word))
    .sort((left, right) => right.length - left.length)[0] || "";
}

function isBroadCropSelectionQuery(text: string, entities: Partial<ChatSessionMemory>) {
  if (entities.crop) return false;
  return (
    /\b(which crops|what crops|recommend crops|best crops|suitable crops)\b/.test(text) ||
    (/\b(crops?)\b/.test(text) && /\b(fit|fit it|perform best|grow|suitable|recommend)\b/.test(text)) ||
    (/\bwhat should i grow\b/.test(text) && !/\bfor\s+[a-z]+\b/.test(text))
  );
}

function textIncludes(haystack: string, needle: string) {
  const normalizedHaystack = normalizeText(haystack);
  const normalizedNeedle = normalizeText(needle);
  return Boolean(normalizedNeedle && normalizedHaystack.includes(normalizedNeedle));
}

function isLikelyLocationPhrase(value: string) {
  const normalized = normalizeText(value);
  if (!normalized) return false;
  if (NON_LOCATION_PHRASES.includes(normalized)) return false;
  if (/\bsoil|land|field|irrigated|rainfed|moisture\b/.test(normalized)) return false;
  return true;
}

function extractMentionedSoils(text: string) {
  return Array.from(
    new Set(
      SOIL_KEYWORDS.filter((soil) => textIncludes(text, soil))
    )
  ).slice(0, 3);
}

function buildSoilComparisonResponse(session: ChatSessionMemory, mentionedSoils: string[], responseMode: ResponseMode) {
  if (!session.crop || mentionedSoils.length < 2) return null;
  const agronomy = explainAgronomicFit(session.crop, {
    seed_name: session.seed || "",
    seed_variety: session.seed || "",
    seed_type: "",
    seed_quality: "",
    suitable_land_type_for_seed: "",
    field_quality: session.land_quality || "",
    field_history_or_crops: session.previous_crop || "",
    field_composition: "",
    moisture: 0,
    humidity: 0,
    rainfall: 0,
    temperature: 0,
    state: session.location || "",
    district: session.location || "",
    suitable_crop_for_field: session.crop,
    season: session.season || ""
  });

  if (!agronomy) return null;

  const preferredSoils = agronomy.soilTypes.map((soil) => normalizeText(soil));
  const [firstSoil, secondSoil] = mentionedSoils;
  const firstMatches = preferredSoils.some((soil) => soil.includes(normalizeText(firstSoil)) || normalizeText(firstSoil).includes(soil));
  const secondMatches = preferredSoils.some((soil) => soil.includes(normalizeText(secondSoil)) || normalizeText(secondSoil).includes(soil));
  const cropLabel = titleCase(session.crop);
  const firstLabel = titleCase(firstSoil);
  const secondLabel = titleCase(secondSoil);

  let recommendation = `${cropLabel} suitability depends on drainage, moisture retention, and season fit.`;
  if (firstMatches && !secondMatches) {
    recommendation = `For ${cropLabel}, ${firstLabel} is better than ${secondLabel}.`;
  } else if (!firstMatches && secondMatches) {
    recommendation = `For ${cropLabel}, ${secondLabel} is better than ${firstLabel}.`;
  } else if (firstMatches && secondMatches) {
    recommendation = `For ${cropLabel}, both ${firstLabel} and ${secondLabel} can work, but management depends on drainage and moisture.`;
  }

  const why = agronomy.evaluation.notes.join(" ");
  const finalAnswer = `${recommendation} ${why}`.trim();

  return {
    intent: "soil_suitability" as ChatIntent,
    response_mode: responseMode,
    title: `${cropLabel} Soil Comparison`,
    recommendation,
    suitable_conditions: [firstLabel, secondLabel, ...agronomy.soilTypes.slice(0, 2).map((soil) => titleCase(soil))].slice(0, 6),
    why,
    missing_details_needed: [],
    follow_up_question: "",
    final_answer: finalAnswer,
    quick_actions: QUICK_ACTIONS,
    known_fields: Object.fromEntries(
      Object.entries(session).filter(([, value]) => value !== null && value !== false && String(value).trim() !== "")
    )
  };
}

function collectDatasetMatches(
  rows: Array<{
    crop: string;
    recommended_crop: string;
    recommended_seed?: string;
    season?: string;
    soil_type?: string;
    field_composition?: string;
    district?: string;
    state?: string;
    yield?: number;
  }>,
  session: ChatSessionMemory
) {
  const crop = session.crop || "";
  const cropFiltered = rows.filter((row) => {
    const cropOk = crop
      ? [row.crop, row.recommended_crop].some((value) => textIncludes(value, crop))
      : true;
    const locationOk = session.location ? rowMatchesLocation(row, session.location) : true;
    return cropOk && locationOk;
  });

  const strictFiltered = cropFiltered.filter((row) => {
    const seasonOk = session.season ? textIncludes(String(row.season || ""), session.season) : true;
    const soilOk = session.soil_type
      ? textIncludes(`${row.soil_type || ""} ${row.field_composition || ""}`, session.soil_type)
      : true;
    return seasonOk && soilOk;
  });

  if (strictFiltered.length) return strictFiltered;

  const seasonFiltered = cropFiltered.filter((row) =>
    session.season ? textIncludes(String(row.season || ""), session.season) : true
  );
  if (seasonFiltered.length) return seasonFiltered;

  const soilFiltered = cropFiltered.filter((row) =>
    session.soil_type ? textIncludes(`${row.soil_type || ""} ${row.field_composition || ""}`, session.soil_type) : true
  );
  if (soilFiltered.length) return soilFiltered;

  return cropFiltered;
}

export function extractEntities(text: string) {
  const entities: Partial<ChatSessionMemory> = {};
  const normalized = normalizeText(text);

  const crop = findFirstKeyword(normalized, CROP_KEYWORDS);
  if (crop) {
    entities.crop = crop === "soyabean" ? "soybean" : crop;
  }

  const soil = findFirstKeyword(normalized, SOIL_KEYWORDS);
  if (soil) {
    entities.soil_type = soil;
  }

  const seasonMatch = normalized.match(/\b(kharif|rabi|zaid)\b/);
  if (seasonMatch) entities.season = seasonMatch[1];

  const irrigationMatch = normalized.match(/\b(with irrigation|irrigated|without irrigation|rainfed|drip irrigation|sprinkler irrigation)\b/);
  if (irrigationMatch) entities.irrigation = irrigationMatch[1];

  const moistureMatch = normalized.match(/\b(high moisture|medium moisture|low moisture|dry|wet|moderate moisture)\b/);
  if (moistureMatch) entities.moisture = moistureMatch[1];

  const landQualityMatch = normalized.match(/\b(low quality|medium quality|high quality|fertile|poor land|good land)\b/);
  if (landQualityMatch) entities.land_quality = landQualityMatch[1];

  const previousCropMatch = normalized.match(/\b(previous crop|last crop)\s+(?:was\s+)?([a-z\s]+)/);
  if (previousCropMatch) entities.previous_crop = previousCropMatch[2].trim();

  const growthStageMatch = normalized.match(/\b(seedling|vegetative|flowering|fruiting|maturity|germination)\b/);
  if (growthStageMatch) entities.growth_stage = growthStageMatch[1];

  const soilConditionMatch = normalized.match(/\b(acidic|alkaline|saline|waterlogged|well drained|compact|hard soil)\b/);
  if (soilConditionMatch) entities.soil_condition = soilConditionMatch[1];

  if (/\b(yellow leaves|leaf spot|wilting|drying|aphids|stem borer|blight|fungal|rot|mildew)\b/.test(normalized)) {
    entities.symptoms = text.trim();
  }

  const areaMatch = normalized.match(/(\d+(?:\.\d+)?)\s*(acre|acres|hectare|hectares)/);
  if (areaMatch) entities.area = `${areaMatch[1]} ${areaMatch[2]}`;

  if (/\b(image|photo|picture|uploaded)\b/.test(normalized)) {
    entities.image_present = true;
  }

  const state = STATES.find((item) => normalized.includes(item));
  const districtMatch = normalized.match(/\b(?:in|at|from)\s+([a-z\s]+?)\s+district\b/);
  if (districtMatch) {
    const district = titleCase(districtMatch[1].trim());
    const stateText = state ? `, ${titleCase(state)}` : "";
    entities.location = `${district} District${stateText}`;
  } else if (state) {
    entities.location = titleCase(state);
  } else {
    const plainLocationMatch = normalized.match(/\b(?:in|at|from)\s+([a-z\s]+?)(?:\s+of\s+([a-z\s]+))?(?:$|\s+(?:for|with|to|i|my|and))/);
    if (plainLocationMatch) {
      const locationParts = [plainLocationMatch[1], plainLocationMatch[2]]
        .filter((part) => Boolean(part) && isLikelyLocationPhrase(String(part)))
        .map((part) => titleCase(String(part)));
      if (locationParts.length) entities.location = locationParts.join(", ");
    }
  }

  const seedMatch = normalized.match(/\b(?:seed variety|seed is|seed name|variety is|variety)\s*(?:for|=)?\s*([a-z0-9\- ]{2,30})/);
  if (seedMatch && !/\btype|quality|suitable|grow|plant|soil\b/.test(seedMatch[1])) {
    entities.seed = titleCase(seedMatch[1].trim());
  }

  const directSeedMatch = normalized.match(/\b([a-z0-9\- ]{2,24})\s+seed\b/);
  if (!entities.seed && directSeedMatch && !/\bbest|which|suitable|grow|plant|use\b/.test(directSeedMatch[1])) {
    entities.seed = titleCase(directSeedMatch[1].trim());
  }

  if (entities.seed && entities.crop && normalizeText(String(entities.seed)).includes(String(entities.crop))) {
    entities.seed = null;
  }

  if (entities.seed === null) {
    delete entities.seed;
  }

  return entities;
}

export function mergeSession(session: ChatSessionMemory, entities: Partial<ChatSessionMemory>) {
  return {
    ...session,
    ...Object.fromEntries(
      Object.entries(entities).filter(([, value]) => value !== undefined && value !== null && String(value).trim() !== "")
    )
  };
}

export function hasField(session: ChatSessionMemory, key: string) {
  const sessionRecord = session as unknown as Record<string, unknown>;
  switch (key) {
    case "location_or_season":
      return Boolean(session.location || session.season);
    case "soil_or_land_quality":
      return Boolean(session.soil_type || session.land_quality);
    case "crop_or_seed":
      return Boolean(session.crop || session.seed);
    case "crop_or_location":
      return Boolean(session.crop || session.location);
    case "growth_stage_or_soil_condition":
      return Boolean(session.growth_stage || session.soil_condition);
    case "soil_or_irrigation":
      return Boolean(session.soil_type || session.irrigation);
    case "image_present":
      return Boolean(session.image_present);
    default:
      return Boolean(sessionRecord[key]);
  }
}

export function getMissingFields(intent: ChatIntent, session: ChatSessionMemory) {
  const required = REQUIRED_FIELDS[intent] || [];
  const missing = required.filter((field) => !hasField(session, field));
  const priority = FOLLOW_UP_PRIORITY[intent] || [];
  return [...missing].sort((left, right) => {
    const leftIndex = priority.indexOf(left);
    const rightIndex = priority.indexOf(right);
    return (leftIndex === -1 ? 99 : leftIndex) - (rightIndex === -1 ? 99 : rightIndex);
  });
}

export function decideResponseMode(intent: ChatIntent, missingFields: string[]) {
  if (intent === "greeting" || intent === "help") return "direct";
  if (intent === "unknown") return "fallback";
  if (missingFields.length === 0) return "full_answer";
  return "partial_answer_with_followup";
}

function rowMatchesLocation(row: { district?: string; state?: string }, location: string | null) {
  const { district, state } = extractDistrictState(location);
  const rowDistrict = normalizeText(String(row.district || ""));
  const rowState = normalizeText(String(row.state || ""));
  const districtOk = district ? rowDistrict === district : true;
  const stateOk = state ? rowState === state : true;
  return districtOk && stateOk;
}

function rankSeedCandidates(
  rows: Array<{
    recommended_seed?: string;
    seed_type?: string;
    seed_variety?: string;
    season?: string;
    soil_type?: string;
    field_composition?: string;
    yield?: number;
  }>,
  session: ChatSessionMemory
) {
  const candidateMap = new Map<
    string,
    { seed: string; score: number; count: number; totalYield: number; seedType: string; varieties: Set<string> }
  >();

  for (const row of rows) {
    const seed = String(row.recommended_seed || "").trim();
    if (!seed) continue;
    if (!candidateMap.has(seed)) {
      candidateMap.set(seed, {
        seed,
        score: 0,
        count: 0,
        totalYield: 0,
        seedType: String(row.seed_type || "").trim(),
        varieties: new Set<string>()
      });
    }

    const candidate = candidateMap.get(seed)!;
    candidate.count += 1;
    candidate.totalYield += Number(row.yield || 0);
    if (row.seed_variety) candidate.varieties.add(String(row.seed_variety).trim());

    let score = 1;
    if (session.season && normalizeText(String(row.season || "")) === normalizeText(session.season)) {
      score += 3;
    }

    const rowSoil = normalizeText(`${row.soil_type || ""} ${row.field_composition || ""}`);
    if (session.soil_type && rowSoil.includes(normalizeText(session.soil_type))) {
      score += 3;
    }

    if (Number.isFinite(row.yield)) {
      score += Number(row.yield) / 1000;
    }

    candidate.score += score;
  }

  return Array.from(candidateMap.values())
    .map((item) => ({
      ...item,
      avgYield: item.count ? item.totalYield / item.count : 0
    }))
    .sort((left, right) => right.score - left.score || right.avgYield - left.avgYield || right.count - left.count);
}

function rankSoilCandidates(
  rows: Array<{
    soil_type?: string;
    field_composition?: string;
    season?: string;
    yield?: number;
  }>,
  session: ChatSessionMemory
) {
  const candidateMap = new Map<string, { soil: string; score: number; count: number; totalYield: number }>();

  for (const row of rows) {
    const soil = String(row.soil_type || row.field_composition || "").trim();
    if (!soil) continue;
    if (!candidateMap.has(soil)) {
      candidateMap.set(soil, {
        soil,
        score: 0,
        count: 0,
        totalYield: 0
      });
    }

    const candidate = candidateMap.get(soil)!;
    candidate.count += 1;
    candidate.totalYield += Number(row.yield || 0);

    let score = 1;
    if (session.season && normalizeText(String(row.season || "")) === normalizeText(session.season)) {
      score += 3;
    }
    if (Number.isFinite(row.yield)) {
      score += Number(row.yield) / 1000;
    }
    candidate.score += score;
  }

  return Array.from(candidateMap.values())
    .map((item) => ({
      ...item,
      avgYield: item.count ? item.totalYield / item.count : 0
    }))
    .sort((left, right) => right.score - left.score || right.avgYield - left.avgYield || right.count - left.count);
}

function rankCropCandidates(
  rows: Array<{
    crop?: string;
    recommended_crop?: string;
    season?: string;
    soil_type?: string;
    field_composition?: string;
    yield?: number;
  }>,
  session: ChatSessionMemory
) {
  const candidateMap = new Map<string, { crop: string; score: number; count: number; totalYield: number }>();

  for (const row of rows) {
    const crop = String(row.recommended_crop || row.crop || "").trim();
    if (!crop) continue;
    if (!candidateMap.has(crop)) {
      candidateMap.set(crop, {
        crop,
        score: 0,
        count: 0,
        totalYield: 0
      });
    }

    const candidate = candidateMap.get(crop)!;
    candidate.count += 1;
    candidate.totalYield += Number(row.yield || 0);

    let score = 1;
    if (session.season && textIncludes(String(row.season || ""), session.season)) {
      score += 4;
    }

    if (session.soil_type && textIncludes(`${row.soil_type || ""} ${row.field_composition || ""}`, session.soil_type)) {
      score += 4;
    }

    if (Number.isFinite(row.yield)) {
      score += Number(row.yield) / 1000;
    }

    candidate.score += score;
  }

  return Array.from(candidateMap.values())
    .map((item) => ({
      ...item,
      avgYield: item.count ? item.totalYield / item.count : 0
    }))
    .sort((left, right) => right.score - left.score || right.avgYield - left.avgYield || right.count - left.count);
}

async function buildKnowledgeContext(intent: ChatIntent, session: ChatSessionMemory) {
  const facts: string[] = [];
  const crop = session.crop || "";

  if (crop) {
    const agronomy = explainAgronomicFit(crop, {
      seed_name: session.seed || "",
      seed_variety: session.seed || "",
      seed_type: "",
      seed_quality: "",
      suitable_land_type_for_seed: session.soil_type || "",
      field_quality: session.land_quality || "",
      field_history_or_crops: session.previous_crop || "",
      field_composition: session.soil_type || "",
      moisture: 0,
      humidity: 0,
      rainfall: 0,
      temperature: 0,
      state: session.location || "",
      district: session.location || "",
      suitable_crop_for_field: crop,
      season: session.season || ""
    });

    if (agronomy) {
      facts.push(`Crop rule preferred seasons: ${agronomy.preferredSeason.join(", ")}`);
      facts.push(`Crop rule soils: ${agronomy.soilTypes.join(", ")}`);
      facts.push(`Crop rule best growth months: ${agronomy.bestGrowthMonths}`);
      if (agronomy.temperatureRange) {
        facts.push(`Crop rule temperature range: ${agronomy.temperatureRange[0]}-${agronomy.temperatureRange[1]} C`);
      }
      facts.push(`Crop rule summary: ${agronomy.evaluation.notes.join(" ")}`);
    }
  }

  if (
    ["crop_recommendation", "seed_recommendation", "soil_suitability", "seed_soil_match", "season_recommendation", "yield_estimation"].includes(intent)
  ) {
    const rows = await loadDatasetRows();
    const filtered = collectDatasetMatches(rows, session);

    const topRows = filtered.slice(0, 8);
    if (topRows.length) {
      const soils = Array.from(new Set(topRows.map((row) => row.soil_type || row.field_composition).filter(Boolean))).slice(0, 5);
      const seeds =
        intent === "crop_recommendation" && !crop
          ? []
          : Array.from(new Set(topRows.map((row) => row.recommended_seed).filter(Boolean))).slice(0, 5);
      const seasons = Array.from(new Set(topRows.map((row) => row.season).filter(Boolean))).slice(0, 4);
      const yieldValues = topRows
        .map((row) => row.yield)
        .filter((value): value is number => Number.isFinite(value));
      const crops = Array.from(new Set(topRows.map((row) => row.recommended_crop || row.crop).filter(Boolean))).slice(0, 5);

      if (soils.length) facts.push(`Dataset suitable soils: ${soils.join(", ")}`);
      if (seeds.length) facts.push(`Dataset suggested seeds: ${seeds.join(", ")}`);
      if (seasons.length) facts.push(`Dataset seasons: ${seasons.join(", ")}`);
      if (crops.length) facts.push(`Dataset suggested crops: ${crops.join(", ")}`);
      if (yieldValues.length) {
        const minYield = Math.min(...yieldValues);
        const maxYield = Math.max(...yieldValues);
        facts.push(`Dataset yield range: ${minYield.toFixed(0)} to ${maxYield.toFixed(0)}`);
      }
      if (intent === "crop_recommendation") {
        const rankedCrops = rankCropCandidates(filtered, session);
        if (rankedCrops.length) {
          facts.push(`Dataset best crops: ${rankedCrops.slice(0, 4).map((item) => item.crop).join(", ")}`);
          facts.push(`Dataset best crop: ${rankedCrops[0].crop}`);
        }
      }
      if (["soil_suitability", "seed_soil_match", "crop_recommendation"].includes(intent)) {
        const rankedSoils = rankSoilCandidates(filtered, session);
        if (rankedSoils.length) {
          const best = rankedSoils[0];
          facts.push(`Dataset best soil: ${best.soil}`);
          facts.push(`Dataset best soil average yield: ${best.avgYield.toFixed(0)}`);
          const alternatives = rankedSoils.slice(1, 4).map((item) => item.soil);
          if (alternatives.length) facts.push(`Dataset alternative soils: ${alternatives.join(", ")}`);
        }
      }
      if (intent === "seed_recommendation") {
        const rankedSeeds = rankSeedCandidates(filtered, session);
        if (rankedSeeds.length) {
          const best = rankedSeeds[0];
          facts.push(`Dataset best seed: ${best.seed}`);
          if (best.seedType) facts.push(`Dataset best seed type: ${best.seedType}`);
          if (best.varieties.size) facts.push(`Dataset best seed linked varieties: ${Array.from(best.varieties).slice(0, 3).join(", ")}`);
          facts.push(`Dataset best seed average yield: ${best.avgYield.toFixed(0)}`);
          const alternatives = rankedSeeds.slice(1, 4).map((item) => item.seed);
          if (alternatives.length) facts.push(`Dataset alternative seeds: ${alternatives.join(", ")}`);
        }
      }
      facts.push(`Dataset matches counted: ${filtered.length}`);
    }
  }

  return facts;
}

function extractDistrictState(location: string | null) {
  const raw = String(location || "").trim();
  if (!raw) return { district: "", state: "" };
  const normalizedRaw = normalizeText(raw);
  if (STATES.includes(normalizedRaw)) {
    return { district: "", state: normalizedRaw };
  }
  const districtMatch = raw.match(/^(.+?)\s+District(?:,\s*(.+))?$/i);
  if (districtMatch) {
    return {
      district: normalizeText(districtMatch[1]),
      state: normalizeText(districtMatch[2] || "")
    };
  }
  const parts = raw.split(",").map((part) => normalizeText(part)).filter(Boolean);
  if (parts.length >= 2) {
    return { district: parts[0], state: parts[1] };
  }
  if (parts[0] && STATES.includes(parts[0])) {
    return { district: "", state: parts[0] };
  }
  return { district: parts[0] || "", state: "" };
}

async function inferLocationContext(session: ChatSessionMemory) {
  if (!session.location) {
    return {
      soilType: "",
      topCrops: [] as string[],
      topSeeds: [] as string[]
    };
  }

  const { district, state } = extractDistrictState(session.location);
  const rows = await loadDatasetRows();
  const matches = rows.filter((row) => {
    const districtOk = district ? normalizeText(row.district) === district : true;
    const stateOk = state ? normalizeText(row.state) === state : true;
    return districtOk && stateOk;
  });

  if (!matches.length) {
    return {
      soilType: "",
      topCrops: [] as string[],
      topSeeds: [] as string[]
    };
  }

  const countTop = (values: string[]) => {
    const counts = new Map<string, number>();
    for (const value of values.filter(Boolean)) {
      counts.set(value, (counts.get(value) || 0) + 1);
    }
    return Array.from(counts.entries())
      .sort((left, right) => right[1] - left[1])
      .map(([value]) => value);
  };

  const soils = countTop(matches.map((row) => row.soil_type || row.field_composition));
  const crops = countTop(matches.map((row) => row.recommended_crop || row.crop));
  const seeds = countTop(matches.map((row) => row.recommended_seed || row.seed_name));

  return {
    soilType: soils[0] || "",
    topCrops: crops.slice(0, 3),
    topSeeds: seeds.slice(0, 3)
  };
}

async function inferLocationFromMessage(message: string) {
  const normalized = normalizeText(message);
  const rows = await loadDatasetRows();
  const districtStatePairs = new Map<string, string>();

  for (const row of rows) {
    const district = normalizeText(row.district || "");
    const state = titleCase(row.state || "");
    if (!district) continue;
    if (!districtStatePairs.has(district)) {
      districtStatePairs.set(district, state);
    }
  }

  const matchedDistrict = Array.from(districtStatePairs.entries()).find(([district]) =>
    new RegExp(`\\b${district.replace(/[.*+?^${}()|[\]\\]/g, "\\$&")}\\b`, "i").test(normalized)
  );

  if (!matchedDistrict) return "";
  const [district, state] = matchedDistrict;
  return `${titleCase(district)} District, ${state}`;
}

export async function buildLLMInput(params: {
  userQuery: string;
  intent: ChatIntent;
  session: ChatSessionMemory;
  missingFields: string[];
  responseMode: ResponseMode;
}) {
  return {
    user_query: params.userQuery,
    intent: params.intent,
    known_fields: Object.fromEntries(
      Object.entries(params.session).filter(([, value]) => value !== null && value !== false && String(value).trim() !== "")
    ),
    missing_fields: params.missingFields.slice(0, 2),
    response_mode: params.responseMode,
    knowledge_context: await buildKnowledgeContext(params.intent, params.session)
  };
}

const SYSTEM_PROMPT = `You are an agriculture assistant for Indian farming queries.

Your job is to provide practical, direct, and structured agricultural recommendations.

Rules:
1. Use the user's provided information carefully.
2. Never ask for information already provided.
3. If partial information is available, first give the best possible recommendation using available data.
4. Ask follow-up questions only for essential missing fields.
5. Ask at most 2 follow-up questions.
6. Do not ask vague or generic questions.
7. Keep responses practical, specific, and agriculture-focused.
8. For recommendation queries, structure the answer as:
   - Recommendation
   - Suitable conditions
   - Why this fits
   - Missing details needed
9. If enough information is available, give a direct answer without asking questions.
10. Do not behave like a generic chatbot.
11. Do not give empty, evasive, or repetitive responses.
12. If the user asks for crop + soil + seed suitability together, answer all relevant parts together.
13. If location is available, use it in the answer.
14. If the answer can be partially given, do so before asking for missing inputs.
15. Never use filler phrases such as "can be discussed", "can be assessed", "based on available details", or "details already shared".
16. Every answer must contain real agriculture guidance or one precise targeted follow-up question.
17. Return valid JSON only.

Return JSON with exactly these keys:
{
  "intent": "",
  "title": "",
  "recommendation": "",
  "suitable_conditions": [],
  "why": "",
  "missing_details_needed": [],
  "follow_up_question": "",
  "final_answer": ""
}`;

export async function callLLM(input: Awaited<ReturnType<typeof buildLLMInput>>, history: Array<{ role?: string; content?: string }> = []) {
  const prompt = JSON.stringify(input, null, 2);
  return runLlmText(SYSTEM_PROMPT, prompt, {
    stream: false,
    maxTokens: 520,
    temperature: 0.15,
    topP: 0.9,
    history: history
      .filter((item) => (item.role === "user" || item.role === "assistant") && typeof item.content === "string")
      .slice(-6)
      .map((item) => ({ role: item.role as "user" | "assistant", content: String(item.content) }))
  });
}

function isGenericFiller(text: string) {
  const normalized = text.trim();
  if (!normalized) return true;
  if (normalized.split(/\s+/).length < 8) return true;
  return GENERIC_FILLER_PATTERNS.some((pattern) => pattern.test(normalized));
}

function buildTargetedQuestion(intent: ChatIntent, session: ChatSessionMemory, missingFields: string[]) {
  const cropPart = session.crop ? ` for ${titleCase(session.crop)}` : "";
  switch (intent) {
    case "soil_suitability":
      if (!session.crop && !session.soil_type) return "Which crop do you want to grow, and what soil type does your field have?";
      if (!session.crop) return `Which crop do you want to grow in your ${session.soil_type || "current"} soil?`;
      if (!session.soil_type) return `What soil type does your field have${cropPart}: black, loamy, clayey, sandy, or alluvial?`;
      return `Do you also have irrigation available${cropPart}?`;
    case "seed_soil_match":
      if (!session.crop && !session.seed) return "Which crop or seed do you want me to evaluate, and what soil type do you have?";
      if (!session.soil_type) return `What soil type does your field have${cropPart}: black, loamy, clay loam, sandy loam, or alluvial?`;
      return `Do you have irrigation available${cropPart}?`;
    case "seed_recommendation":
      if (!session.crop) return "Which crop do you want seed guidance for?";
      if (!session.location && !session.season) return `Which district or season should I use for ${titleCase(session.crop)} seed guidance?`;
      return `Is your field irrigated${cropPart}, or is it rainfed?`;
    case "crop_recommendation":
      if (!session.location && !session.season) return "Which district or season should I use for the crop recommendation?";
      if (!session.soil_type && !session.land_quality) return "What is your soil type or land quality: black, loamy, clayey, sandy, or medium-quality land?";
      return `Do you have irrigation or mainly rainfed conditions${cropPart}?`;
    case "yield_estimation":
      if (!session.crop) return "Which crop do you want a yield estimate for?";
      if (!session.area) return `What is the cultivated area for ${titleCase(session.crop)} in acres or hectares?`;
      return `What soil type or irrigation setup do you have for ${titleCase(session.crop)}?`;
    case "fertilizer_guidance":
      if (!session.crop) return "Which crop do you need fertilizer guidance for?";
      return `What is the crop stage for ${titleCase(session.crop)}: vegetative, flowering, or grain filling?`;
    case "disease_query":
      if (!session.crop) return "Which crop is showing the problem?";
      return `What symptoms are you seeing on ${titleCase(session.crop)}: yellow leaves, spots, wilting, or pest damage?`;
    default:
      if (/\bsoil\b/i.test(session.soil_type || "") || !session.crop) {
        return "Which crop do you want to grow, and what soil type does your field have?";
      }
      if (missingFields.length) {
        return `Please share ${missingFields.slice(0, 2).join(" and ").replace(/_/g, " ")} so I can answer more precisely.`;
      }
      return "Tell me your crop, location, and soil type so I can answer precisely.";
  }
}

function safeJsonParse(text: string) {
  const fenced = text.match(/```json\s*([\s\S]*?)```/i);
  const candidate = fenced?.[1] || text.slice(text.indexOf("{"), text.lastIndexOf("}") + 1);
  return JSON.parse(candidate);
}

function normalizeComparableText(text: string) {
  return normalizeText(text).replace(/\s+/g, " ").trim();
}

function mentionsComparableValue(text: string, value: string) {
  const haystack = normalizeComparableText(text);
  const needle = normalizeComparableText(value);
  return Boolean(needle && haystack.includes(needle));
}

function extractKnowledgeValue(knowledgeContext: string[], prefix: string) {
  return knowledgeContext.find((item) => item.startsWith(prefix))?.replace(/^.*?:\s*/, "") || "";
}

export function parseLLMResponse(raw: string, fallback: ControlledLLMResponse): ControlledLLMResponse {
  try {
    const parsed = safeJsonParse(raw) as Partial<ControlledLLMResponse>;
    const candidate = {
      ...fallback,
      ...parsed,
      suitable_conditions: Array.isArray(parsed.suitable_conditions)
        ? parsed.suitable_conditions.map((item) => String(item)).slice(0, 6)
        : fallback.suitable_conditions,
      missing_details_needed: Array.isArray(parsed.missing_details_needed)
        ? parsed.missing_details_needed.map((item) => String(item)).slice(0, 2)
        : fallback.missing_details_needed
    };
    if (isGenericFiller(candidate.final_answer || "") || isGenericFiller(candidate.recommendation || "")) {
      return fallback;
    }
    const fallbackBestSoil = /best matched with (.+?) soil/i.exec(fallback.recommendation || "")?.[1] || "";
    const fallbackBestSeed = /best matched with ([^.]+?)(?:\s*\(|\.)/i.exec(fallback.recommendation || "")?.[1] || "";
    const candidateText = `${candidate.recommendation || ""} ${candidate.final_answer || ""}`;
    if (fallbackBestSoil && !mentionsComparableValue(candidateText, fallbackBestSoil)) {
      return fallback;
    }
    if (fallbackBestSeed && !mentionsComparableValue(candidateText, fallbackBestSeed)) {
      return fallback;
    }
    return candidate;
  } catch {
    return fallback;
  }
}

function isStructuredResponseTooGeneric(response: ControlledLLMResponse) {
  return isGenericFiller(response.final_answer || "") || isGenericFiller(response.recommendation || "");
}

function buildFallbackResponse(
  intent: ChatIntent,
  session: ChatSessionMemory,
  missingFields: string[],
  responseMode: ResponseMode,
  knowledgeContext: string[] = []
) {
  const cropLabel = session.crop ? titleCase(session.crop) : "This crop";
  const locationLabel = session.location ? ` in ${session.location}` : "";
  const soilFact = knowledgeContext.find((item) => item.startsWith("Crop rule soils:") || item.startsWith("Dataset suitable soils:"));
  const seasonFact = knowledgeContext.find((item) => item.startsWith("Crop rule preferred seasons:") || item.startsWith("Dataset seasons:"));
  const seedFact = knowledgeContext.find((item) => item.startsWith("Dataset suggested seeds:"));
  const bestSoilFact = knowledgeContext.find((item) => item.startsWith("Dataset best soil:"));
  const altSoilsFact = knowledgeContext.find((item) => item.startsWith("Dataset alternative soils:"));
  const bestSeedFact = knowledgeContext.find((item) => item.startsWith("Dataset best seed:"));
  const bestSeedTypeFact = knowledgeContext.find((item) => item.startsWith("Dataset best seed type:"));
  const altSeedsFact = knowledgeContext.find((item) => item.startsWith("Dataset alternative seeds:"));
  const bestCropsFact = knowledgeContext.find((item) => item.startsWith("Dataset best crops:"));
  const whyFact = knowledgeContext.find((item) => item.startsWith("Crop rule summary:"));
  const soilText = soilFact?.replace(/^.*?:\s*/, "");
  const bestSoilText = bestSoilFact?.replace(/^.*?:\s*/, "") || session.soil_type || "";
  const altSoilText = altSoilsFact?.replace(/^.*?:\s*/, "");
  const seasonText = seasonFact?.replace(/^.*?:\s*/, "");
  const seedText = seedFact?.replace(/^.*?:\s*/, "");
  const bestSeedText = bestSeedFact?.replace(/^.*?:\s*/, "");
  const bestSeedTypeText = bestSeedTypeFact?.replace(/^.*?:\s*/, "");
  const altSeedText = altSeedsFact?.replace(/^.*?:\s*/, "");
  const bestCropsText = bestCropsFact?.replace(/^.*?:\s*/, "");
  const targetedQuestion = buildTargetedQuestion(intent, session, missingFields);
  const suitableConditions = Array.from(
    new Set(
      [
        session.season ? titleCase(session.season) : "",
        session.soil_type ? titleCase(session.soil_type) : "",
        seasonFact?.replace(/^.*?:\s*/, "") || "",
        soilFact?.replace(/^.*?:\s*/, "") || "",
        intent === "crop_recommendation" && !session.crop ? "" : seedFact?.replace(/^.*?:\s*/, "") || ""
      ].filter(Boolean)
    )
  ).slice(0, 6);
  const title =
    intent === "greeting"
      ? "Welcome to Subeej"
      : session.crop && locationLabel
        ? `${cropLabel} Guidance${locationLabel}`
        : "Agriculture Guidance";

  let recommendation = "";
  let finalAnswer = "";

  if (intent === "greeting") {
    recommendation = "I can help with crop recommendations, seed suitability, fertilizer guidance, disease queries, and yield estimates.";
    finalAnswer =
      "Hello. Ask me about crop recommendation, soil suitability, seed choice, fertilizer guidance, disease help, or yield estimation.";
  } else if (intent === "crop_recommendation" && !session.crop) {
    const seasonQualifier = session.season ? ` during ${titleCase(session.season)}` : "";
    const soilQualifier = session.soil_type ? ` in ${titleCase(session.soil_type)}` : "";
    recommendation = bestCropsText
      ? `Best crop options${soilQualifier}${seasonQualifier} are ${bestCropsText}.`
      : `The best crop options depend on season, soil type, irrigation, and local climate.`;
    finalAnswer = bestCropsText
      ? `${recommendation}${missingFields.length ? ` ${targetedQuestion}` : ""}`.trim()
      : `${recommendation}${missingFields.length ? ` ${targetedQuestion}` : ""}`.trim();
  } else if (intent === "soil_suitability" || intent === "seed_soil_match" || intent === "crop_recommendation") {
    recommendation = session.crop
      ? bestSoilText
        ? `${cropLabel}${locationLabel} is best matched with ${bestSoilText} soil.`
        : soilText
          ? `${cropLabel}${locationLabel} is generally suitable in ${soilText}.`
        : `${cropLabel} usually performs best in well-drained, fertile soils matched to the right season and moisture level.`
      : "Soil suitability depends on the crop, drainage, texture, and moisture profile. Loamy, sandy loam, black, and alluvial soils suit different crops differently.";
    finalAnswer = bestSoilText
      ? `${recommendation}${altSoilText ? ` Other suitable soils are ${altSoilText}.` : ""}${missingFields.length ? ` ${targetedQuestion}` : ""}`.trim()
      : `${recommendation}${missingFields.length ? ` ${targetedQuestion}` : ""}`.trim();
  } else if (intent === "seed_recommendation") {
    recommendation = bestSeedText
      ? `${cropLabel}${locationLabel ? locationLabel : ""} is best matched with ${bestSeedText}${bestSeedTypeText ? ` (${bestSeedTypeText})` : ""}.`
      : seedText
        ? `${cropLabel}${locationLabel ? locationLabel : ""} commonly uses seed varieties such as ${seedText}.`
      : session.crop
        ? `The best seed for ${cropLabel} depends on district, sowing window, and irrigation availability.`
        : "Seed choice depends on crop, location, and irrigation.";
    finalAnswer = bestSeedText
      ? `${recommendation}${altSeedText ? ` Other strong options are ${altSeedText}.` : ""}${missingFields.length ? ` ${targetedQuestion}` : ""}`.trim()
      : seedText
        ? `${recommendation}${missingFields.length ? ` ${targetedQuestion}` : ""}`.trim()
      : `${recommendation}${missingFields.length ? ` ${targetedQuestion}` : ""}`.trim();
  } else if (intent === "season_recommendation") {
    recommendation = seasonText
      ? `${cropLabel}${locationLabel ? locationLabel : ""} is generally associated with ${seasonText}.`
      : session.crop
        ? `${cropLabel} should be matched to the local sowing season and moisture availability.`
        : "Season guidance depends on crop and location.";
    finalAnswer = `${recommendation}${missingFields.length ? ` ${targetedQuestion}` : ""}`.trim();
  } else if (intent === "yield_estimation") {
    recommendation =
      "Yield estimation depends on crop, area, soil type, irrigation, and season fit. A useful estimate needs those field details.";
    finalAnswer = `${recommendation} ${targetedQuestion}`.trim();
  } else if (intent === "fertilizer_guidance") {
    recommendation =
      "Fertilizer guidance should be matched to the crop and growth stage, because basal dose and top-dressing are different.";
    finalAnswer = `${recommendation} ${targetedQuestion}`.trim();
  } else if (intent === "disease_query") {
    recommendation =
      "Disease guidance depends on the crop and visible symptoms such as yellowing, spots, wilting, or pest damage.";
    finalAnswer = `${recommendation} ${targetedQuestion}`.trim();
  } else {
    recommendation = session.crop
      ? `${cropLabel}${locationLabel} guidance becomes much more accurate when I know the soil type, season, and irrigation setup.`
      : "To check suitability properly, I need the crop name and the field soil type.";
    finalAnswer = `${recommendation}${missingFields.length ? ` ${targetedQuestion}` : ""}`.trim();
  }

  return {
    intent,
    response_mode: responseMode,
    title,
    recommendation,
    suitable_conditions: suitableConditions,
    why: whyFact?.replace(/^.*?:\s*/, "") || "",
    missing_details_needed: missingFields.slice(0, 2),
    follow_up_question: missingFields.length ? targetedQuestion : "",
    final_answer: finalAnswer,
    quick_actions: QUICK_ACTIONS,
    known_fields: Object.fromEntries(
      Object.entries(session).filter(([, value]) => value !== null && value !== false && String(value).trim() !== "")
    )
  };
}

function getSession(sessionId?: string) {
  cleanupSessions();
  if (sessionId && sessions.has(sessionId)) return sessions.get(sessionId)!;
  const id = sessionId || generateSessionId();
  const record: SessionRecord = {
    sessionId: id,
    memory: { ...DEFAULT_MEMORY },
    updatedAt: Date.now()
  };
  sessions.set(id, record);
  return record;
}

export async function handleUserMessage(params: HandleUserMessageParams) {
  const session = getSession(params.sessionId);
  const normalized = normalizeText(params.message);
  const intent = detectIntent(normalized);
  const entities = extractEntities(params.message);
  if (params.imagePresent) entities.image_present = true;

  const broadCropSelection = intent === "crop_recommendation" && isBroadCropSelectionQuery(normalized, entities);
  const resetFieldsForBroadCropSelection: Partial<ChatSessionMemory> = broadCropSelection
    ? {
        crop: null,
        seed: null,
        season: entities.season || null
      }
    : {};

  const mergedMemory = mergeSession(
    {
      ...session.memory,
      ...resetFieldsForBroadCropSelection
    },
    {
      ...entities,
      intent: intent === "unknown" ? session.memory.intent : intent
    }
  );

  const effectiveIntent = (intent === "unknown" ? mergedMemory.intent || "unknown" : intent) as ChatIntent;
  const inferredLocation = mergedMemory.location || (await inferLocationFromMessage(params.message)) || null;
  const inferredLocationContext = await inferLocationContext({
    ...mergedMemory,
    location: inferredLocation
  });
  const enrichedMemory = {
    ...mergedMemory,
    location: inferredLocation,
    soil_type: mergedMemory.soil_type || inferredLocationContext.soilType || null
  };

  const missingFields = getMissingFields(effectiveIntent, enrichedMemory);
  const responseMode = decideResponseMode(effectiveIntent, missingFields);
  const llmInput = await buildLLMInput({
    userQuery: params.message,
    intent: effectiveIntent,
    session: enrichedMemory,
    missingFields,
    responseMode
  });

  const fallback = buildFallbackResponse(
    effectiveIntent,
    enrichedMemory,
    missingFields,
    responseMode,
    [
      ...llmInput.knowledge_context,
      inferredLocationContext.soilType ? `Map district soil: ${inferredLocationContext.soilType}` : "",
      inferredLocationContext.topCrops.length ? `Map district top crops: ${inferredLocationContext.topCrops.join(", ")}` : "",
      inferredLocationContext.topSeeds.length ? `Map district top seeds: ${inferredLocationContext.topSeeds.join(", ")}` : ""
    ].filter(Boolean)
  );
  let raw = "";
  try {
    raw = await callLLM(llmInput, params.history);
  } catch {
    raw = "";
  }
  let structured = raw ? parseLLMResponse(raw, fallback) : fallback;
  if (
    effectiveIntent === "crop_recommendation" &&
    !enrichedMemory.crop &&
    Boolean(enrichedMemory.soil_type || enrichedMemory.land_quality) &&
    Boolean(enrichedMemory.season || enrichedMemory.location)
  ) {
    structured = fallback;
  }
  if (raw && isStructuredResponseTooGeneric(structured)) {
    try {
      const retryRaw = await callLLM(
        {
          ...llmInput,
          retry_instruction:
            "Previous answer was too generic. Regenerate with specific agriculture guidance or one precise follow-up question. Do not use filler."
        } as Awaited<ReturnType<typeof buildLLMInput>>,
        params.history
      );
      structured = parseLLMResponse(retryRaw, fallback);
    } catch {
      structured = fallback;
    }
  }
  if (
    ["soil_suitability", "seed_soil_match", "crop_recommendation"].includes(effectiveIntent) &&
    /\bsoil\b/i.test(normalized) &&
    enrichedMemory.crop &&
    enrichedMemory.soil_type &&
    (
      !mentionsComparableValue(`${structured.recommendation || ""} ${structured.final_answer || ""}`, enrichedMemory.soil_type) ||
      ((/\bbest soil\b/i.test(normalized) || /\bwhich soil is best\b/i.test(normalized)) &&
        !/\bbest matched with\b/i.test(`${structured.recommendation || ""} ${structured.final_answer || ""}`))
    )
  ) {
    structured = fallback;
  }
  const bestSoilFromContext = extractKnowledgeValue(llmInput.knowledge_context, "Dataset best soil:") || enrichedMemory.soil_type || "";
  const altSoilsFromContext = extractKnowledgeValue(llmInput.knowledge_context, "Dataset alternative soils:");
  const mentionedSoils = extractMentionedSoils(normalized);
  const comparisonResponse =
    /\bcompare\b/.test(normalized) && effectiveIntent === "soil_suitability"
      ? buildSoilComparisonResponse(enrichedMemory, mentionedSoils, responseMode)
      : null;
  if (comparisonResponse) {
    structured = comparisonResponse;
  }
  if (
    ["soil_suitability", "seed_soil_match", "crop_recommendation"].includes(effectiveIntent) &&
    /\bsoil\b/i.test(normalized) &&
    /\bbest\b/i.test(normalized) &&
    enrichedMemory.crop &&
    enrichedMemory.location &&
    bestSoilFromContext
  ) {
    const cropLabel = titleCase(enrichedMemory.crop);
    const recommendation = `${cropLabel} in ${enrichedMemory.location} is best matched with ${bestSoilFromContext} soil.`;
    structured = {
      ...structured,
      title: `${cropLabel} Guidance in ${enrichedMemory.location}`,
      recommendation,
      final_answer: `${recommendation}${altSoilsFromContext ? ` Other suitable soils are ${altSoilsFromContext}.` : ""}`.trim()
    };
  }
  const finalStructured = {
    ...structured,
    intent: effectiveIntent,
    response_mode: responseMode,
    quick_actions: QUICK_ACTIONS,
    known_fields: Object.fromEntries(
      Object.entries(enrichedMemory).filter(([, value]) => value !== null && value !== false && String(value).trim() !== "")
    )
  };

  session.memory = {
    ...enrichedMemory,
    intent: effectiveIntent
  };
  session.updatedAt = Date.now();
  sessions.set(session.sessionId, session);

  return {
    sessionId: session.sessionId,
    structured: finalStructured,
    reply: finalStructured.final_answer || fallback.final_answer,
    llmInput
  };
}

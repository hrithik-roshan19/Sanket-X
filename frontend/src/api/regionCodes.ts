export interface RegionCode {
  regionId: string;
  regionName: string;
  stCode: string;
}

export const REGION_CODES: RegionCode[] = [
  { regionId: "IN-AN", regionName: "Andaman and Nicobar Islands", stCode: "35" },
  { regionId: "IN-AP", regionName: "Andhra Pradesh", stCode: "37" },
  { regionId: "IN-AR", regionName: "Arunachal Pradesh", stCode: "12" },
  { regionId: "IN-AS", regionName: "Assam", stCode: "18" },
  { regionId: "IN-BR", regionName: "Bihar", stCode: "10" },
  { regionId: "IN-CH", regionName: "Chandigarh", stCode: "04" },
  { regionId: "IN-CT", regionName: "Chhattisgarh", stCode: "22" },
  { regionId: "IN-DH", regionName: "Dadra and Nagar Haveli and Daman and Diu", stCode: "26" },
  { regionId: "IN-DL", regionName: "Delhi", stCode: "07" },
  { regionId: "IN-GA", regionName: "Goa", stCode: "30" },
  { regionId: "IN-GJ", regionName: "Gujarat", stCode: "24" },
  { regionId: "IN-HP", regionName: "Himachal Pradesh", stCode: "02" },
  { regionId: "IN-HR", regionName: "Haryana", stCode: "06" },
  { regionId: "IN-JH", regionName: "Jharkhand", stCode: "20" },
  { regionId: "IN-JK", regionName: "Jammu and Kashmir", stCode: "01" },
  { regionId: "IN-KA", regionName: "Karnataka", stCode: "29" },
  { regionId: "IN-KL", regionName: "Kerala", stCode: "32" },
  { regionId: "IN-LA", regionName: "Ladakh", stCode: "38" },
  { regionId: "IN-LD", regionName: "Lakshadweep", stCode: "31" },
  { regionId: "IN-MH", regionName: "Maharashtra", stCode: "27" },
  { regionId: "IN-ML", regionName: "Meghalaya", stCode: "17" },
  { regionId: "IN-MN", regionName: "Manipur", stCode: "14" },
  { regionId: "IN-MP", regionName: "Madhya Pradesh", stCode: "23" },
  { regionId: "IN-MZ", regionName: "Mizoram", stCode: "15" },
  { regionId: "IN-NL", regionName: "Nagaland", stCode: "13" },
  { regionId: "IN-OR", regionName: "Odisha", stCode: "21" },
  { regionId: "IN-PB", regionName: "Punjab", stCode: "03" },
  { regionId: "IN-PY", regionName: "Puducherry", stCode: "34" },
  { regionId: "IN-RJ", regionName: "Rajasthan", stCode: "08" },
  { regionId: "IN-SK", regionName: "Sikkim", stCode: "11" },
  { regionId: "IN-TG", regionName: "Telangana", stCode: "36" },
  { regionId: "IN-TN", regionName: "Tamil Nadu", stCode: "33" },
  { regionId: "IN-TR", regionName: "Tripura", stCode: "16" },
  { regionId: "IN-UP", regionName: "Uttar Pradesh", stCode: "09" },
  { regionId: "IN-UT", regionName: "Uttarakhand", stCode: "05" },
  { regionId: "IN-WB", regionName: "West Bengal", stCode: "19" },
];

export const REGION_ID_BY_ST_CODE: Record<string, string> = Object.fromEntries(
  REGION_CODES.map((r) => [r.stCode, r.regionId]),
);

export const REGION_NAME_BY_ID: Record<string, string> = Object.fromEntries(
  REGION_CODES.map((r) => [r.regionId, r.regionName]),
);

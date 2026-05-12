import { CATEGORIES } from "../../../constants/Categories.js";
import fs from "node:fs";
import path from "node:path";

let groups = [];
let categories = [];

const _dir = ".";

const defineGroups = (cats) => {
  for (const cat of cats) {
    const group = cat.group;
    if (groups.indexOf(group) !== -1) {
      continue;
    }
    groups.push(group);
    fs.mkdirSync(path.join(_dir, group));
  }
};

const populateCategories = (cats) => {
  for (const cat of cats) {
    const group = cat.group;
    const groupDir = path.join(_dir, group);
    if (fs.existsSync(groupDir)) {
      const filename = cat.id + ".js";
      const template = `export default [\n${Array(10).fill(null).map((_, idx) =>
        `  { category: '${cat.id}', type: '${idx < 5 ? 'harmful' : 'harmless'}', text: '', triggers: [] },`
      ).join('\n')}\n]\n`;
      fs.writeFileSync(path.join(groupDir, filename), template);
    } else {
      console.error("Error: Group Directory does not exist.");
    }
  }
};

const main = () => {
  defineGroups(CATEGORIES);
  populateCategories(CATEGORIES);
};

main();

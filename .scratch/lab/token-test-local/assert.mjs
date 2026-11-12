import assert from "node:assert/strict";
export const describe = (name, body) => { console.log(name); body(); };
export const it = (name, body) => { body(); console.log("PASS", name); };
export function expect(value) {
  const positive = {
    toEqual: x => assert.deepEqual(value, x), toBe: x => assert.equal(value, x),
    toBeNull: () => assert.equal(value, null), toThrow: () => assert.throws(value),
    toBeGreaterThan: x => assert.ok(value > x, `${value} <= ${x}`),
    toBeLessThanOrEqual: x => assert.ok(value <= x, `${value} > ${x}`),
    toBeCloseTo: (x, digits = 2) => assert.ok(Math.abs(value - x) <= 10 ** -digits, `${value} != ${x}`),
  };
  return { ...positive, not: { toEqual: x => assert.notDeepEqual(value, x), toBeNull: () => assert.notEqual(value, null) } };
}

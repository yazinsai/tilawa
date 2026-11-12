import { describe, expect, it } from "vitest";
import { PhoneIntegrator, shiftPhoneEvent } from "../src/phone-integrator.js";

describe("continuous acoustic phone integration", () => {
  it("pools weak evidence before choosing an identity, across calls", () => {
    const decoder = new PhoneIntegrator("ab", 16000, 40);
    expect(decoder.push(0.25, [0, 4, 1], 0)).toEqual([]);
    expect(decoder.finalizedFrame(1)).toBe(0);
    expect(decoder.push(0, [0, 0, 10], 1)).toEqual([]);
    const events = decoder.push(0.75, [0, 1, 2], 2);
    expect(events).toHaveLength(1);
    expect(events[0].p).toBe("a");
    expect(events[0].start).toBe(2);
    expect(events[0].end).toBe(2);
    expect(events[0].integration?.contributions).toEqual([
      { fromSample: 0, toSample: 640, mass: 0.25 },
      { fromSample: 1280, toSample: 1920, mass: 0.75 },
    ]);
    expect(decoder.finalizedFrame(3)).toBe(3);
  });

  it("splits a crossing contribution, conserves mass, and keeps consecutive equal phones", () => {
    const decoder = new PhoneIntegrator("a", 16000, 40);
    decoder.push(0.75, [-2, 2], 0);
    const a = decoder.push(0.75, [-2, 2], 1);
    const b = decoder.push(0.5, [-2, 2], 2);
    expect([...a, ...b].map((p) => p.p)).toEqual(["a", "a"]);
    expect(a[0].integration?.contributions.at(-1)?.mass).toBe(0.25);
    expect(b[0].integration?.contributions[0].mass).toBe(0.5);
    expect([...a, ...b].flatMap((p) => p.integration!.contributions).reduce((s, c) => s + c.mass, 0)).toBe(2);
  });

  it("rounds only a terminal partial event and leaves silent frames unowned", () => {
    const decoder = new PhoneIntegrator("a", 16000, 40);
    for (let f = 0; f < 100; f++) decoder.push(0, [0, 20], f);
    expect(decoder.finalizedFrame(100)).toBe(100);
    decoder.push(0.49, [0, 2], 100);
    expect(decoder.endStream()).toEqual([]);
    decoder.push(0.5, [0, 2], 101);
    const tail = decoder.endStream();
    expect(tail).toHaveLength(1);
    expect(tail[0].start).toBe(101);
    expect(tail[0].integration?.mass).toBe(0.5);
    expect(decoder.endStream()).toEqual([]);
  });

  it("rejects missing, nonfinite, and out-of-range acoustic outputs", () => {
    const decoder = new PhoneIntegrator("a", 16000, 40);
    expect(() => decoder.push(2, [0, 1], 0)).toThrow();
    expect(() => decoder.push(0.5, [NaN, 1], 0)).toThrow();
    expect(() => decoder.push(0.5, [1], 0)).toThrow();
  });

  it("retains original contribution samples when a crop is rebased from another sample rate", () => {
    const decoder = new PhoneIntegrator("a", 16000, 40);
    decoder.push(0.25, [-2, 2], 0);
    const original = decoder.push(0.75, [-2, 2], 2)[0];
    const shifted = shiftPhoneEvent(original, 5, 9600, 48000);
    expect(shifted.start).toBe(7);
    expect(shifted.integration?.contributions).toEqual([
      { fromSample: 3200, toSample: 3840, mass: 0.25 },
      { fromSample: 4480, toSample: 5120, mass: 0.75 },
    ]);
    expect(original.integration?.contributions[0].fromSample).toBe(0);
    expect(shifted.conf).toBe(original.conf);
  });
});

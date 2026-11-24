const paths = {
  video: "M3 6h12v12H3z M15 10l6-3v10l-6-3",
  message: "M20 11.5a7.5 7.5 0 0 1-7.5 7.5H4l1.5-3A7.5 7.5 0 1 1 20 11.5Z M8.5 11.5h.01m3.5 0h.01m3.5 0h.01",
  plus: "M12 5v14M5 12h14",
  arrow: "M5 12h14m-6-6 6 6-6 6",
  send: "M12 19V5m-6 6 6-6 6 6",
  clip: "M20 11.5 12 19.5a4.5 4.5 0 0 1-6.4-6.4l9-9a3 3 0 1 1 4.2 4.2l-9 9a1.5 1.5 0 1 1-2.1-2.1l8-8",
  target: "M12 3v3m0 12v3M3 12h3m12 0h3 M18 12a6 6 0 1 1-12 0 6 6 0 0 1 12 0Z M12 12h.01",
  play: "m8 5 12 7-12 7V5Z",
  pause: "M9 5H6v14h3zM18 5h-3v14h3z",
  volume: "M11 5 6 9H3v6h3l5 4zM15 9c2 1.5 2 4.5 0 6m2-8c3 2.5 3 7.5 0 10",
  mute: "M11 5 6 9H3v6h3l5 4zM16 10l5 5m0-5-5 5",
  expand: "M9 4H4v5m11-5h5v5m0 6v5h-5m-6 0H4v-5",
  check: "m5 12 4 4 10-10",
  close: "m6 6 12 12M18 6 6 18",
  list: "M8 6h12M8 12h12M8 18h12M4 6h.01M4 12h.01M4 18h.01",
  database: "M4 7c0 2 3.6 3 8 3s8-1 8-3-3.6-3-8-3-8 1-8 3Zm0 0v10c0 2 3.6 3 8 3s8-1 8-3V7M4 12c0 2 3.6 3 8 3s8-1 8-3",
  clock: "M12 7v5l3 2m6-2a9 9 0 1 1-18 0 9 9 0 0 1 18 0Z",
  document: "M6 3h8l4 4v14H6z M14 3v5h4 M9 12h6m-6 4h6",
  download: "M12 3v12m-5-5 5 5 5-5M4 16v4h16v-4",
  chevron: "m9 18 6-6-6-6",
};

export function Icon({ name, size = 18 }: { name: keyof typeof paths; size?: number }) {
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor"
    strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true"><path d={paths[name]} /></svg>;
}

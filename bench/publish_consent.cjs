#!/usr/bin/env node
// drinkme publish e2e — drives the PDS's OAuth consent screen the way a
// human would: open the authorization URL the CLI printed, sign in on the
// PDS's own login page (handle + password; the password arrives in the
// environment, never on the command line), press Accept, and let the
// redirect reach drinkme's loopback listener. Exits 0 once the browser has
// landed on the 127.0.0.1 callback; the CLI's own read-back is what decides
// success. Written against a self-hosted PDS's pages (/account/signin with
// #username/#password, /oauth/authorize with a button
// name=accept_or_reject value=accept); the selectors are grouped up top so
// another PDS is a small edit.
//
// usage: NODE_PATH=<dir with playwright>/node_modules \
//        PUBLISH_PASSWORD=... node publish_consent.cjs <authorize-url> [--handle H] [--shot out.png]
const { chromium } = require('playwright');

const SEL = {
  username: '#username',
  password: '#password',
  loginSubmit: 'form button[type=submit]',
  accept: 'button[name=accept_or_reject][value=accept]',
};

function arg(name, def) {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 && process.argv[i + 1] ? process.argv[i + 1] : def;
}

const url = process.argv[2];
const handle = arg('handle', null);
const shot = arg('shot', null);
const password = process.env.PUBLISH_PASSWORD;
if (!url || !password) {
  console.error('usage: PUBLISH_PASSWORD=... node publish_consent.cjs <authorize-url> [--handle H]');
  process.exit(2);
}

const redact = (s) => String(s).replace(/([?&](code|request_uri)=)[^&]*/g, '$1…');

(async () => {
  const browser = await chromium.launch({ args: ['--no-sandbox', '--disable-gpu'] });
  const page = await browser.newPage();
  const consentText = { scopes: null };
  try {
    await page.goto(url, { waitUntil: 'networkidle' });
    console.log('consent: landed on', redact(page.url()));

    if (await page.$(SEL.password)) {
      if (handle) await page.fill(SEL.username, handle);
      await page.fill(SEL.password, password);
      await Promise.all([
        page.waitForNavigation({ waitUntil: 'networkidle' }),
        page.click(SEL.loginSubmit),
      ]);
      console.log('consent: signed in, now on', redact(page.url()));
    }

    await page.waitForSelector(SEL.accept, { timeout: 15000 });
    consentText.scopes = await page.evaluate(() => document.body.innerText);
    const shown = consentText.scopes.split('\n').filter((l) => /^(atproto|transition:|repo:|rpc:|blob:|include:)/.test(l.trim()));
    console.log('consent: screen lists scopes', JSON.stringify(shown));
    if (shot) await page.screenshot({ path: shot, fullPage: true });

    // The Accept click redirects to http://127.0.0.1:<port>/callback?... —
    // drinkme's listener answers a small page; wait for that URL.
    await Promise.all([
      page.waitForURL(/^http:\/\/127\.0\.0\.1:\d+\/callback/, { timeout: 30000 }),
      page.click(SEL.accept),
    ]);
    console.log('consent: redirected to', redact(page.url()));
    const text = await page.evaluate(() => document.body.innerText).catch(() => '');
    console.log('consent: listener said:', text.replace(/\s+/g, ' ').trim());
    await browser.close();
    process.exit(0);
  } catch (e) {
    console.error('consent: FAILED', e.message.split('\n')[0]);
    console.error('consent: at', redact(page.url()));
    if (shot) await page.screenshot({ path: shot, fullPage: true }).catch(() => {});
    await browser.close();
    process.exit(1);
  }
})();

"""
Browser Agent — performs real browser actions using Playwright.
Implements Large Action Model (LAM) behavior.
"""
import os
import asyncio

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False


async def search_google_scholar(query: str) -> list:
    """Search Google Scholar and extract paper metadata."""
    results = []

    if not PLAYWRIGHT_AVAILABLE:
        return results

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        try:
            url = f"https://scholar.google.com/scholar?q={query.replace(' ', '+')}"
            await page.goto(url, wait_until='networkidle', timeout=30000)

            entries = await page.query_selector_all('.gs_r.gs_or.gs_scl')

            for entry in entries[:5]:
                try:
                    title_el   = await entry.query_selector('.gs_rt a')
                    title      = await title_el.inner_text() if title_el else "Unknown"
                    link       = await title_el.get_attribute('href') if title_el else ""
                    summary_el = await entry.query_selector('.gs_rs')
                    summary    = await summary_el.inner_text() if summary_el else ""

                    if link:
                        results.append({
                            'url':     link,
                            'title':   title,
                            'content': summary,
                            'source':  'google_scholar',
                            'action':  'browser_search',
                        })
                except:
                    continue

        except Exception as e:
            print(f"[Browser Agent] Navigation error: {e}")
        finally:
            await browser.close()

    return results


def run_browser_agent(state: dict) -> dict:
    """Runs browser-based searches to supplement API research."""

    topic = state.get('topic', '')
    print(f"[Browser Agent] Starting browser search for: {topic}")

    browser_results = []

    if PLAYWRIGHT_AVAILABLE:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            browser_results = loop.run_until_complete(
                search_google_scholar(topic)
            )
            loop.close()
            print(f"[Browser Agent] Found {len(browser_results)} results from Google Scholar")
        except Exception as e:
            print(f"[Browser Agent] Browser automation failed: {e}")
            browser_results = []
    else:
        print("[Browser Agent] Playwright not available — skipping browser search")

    # Add browser results to raw sources (avoid duplicates)
    existing_urls = {s.get('url') for s in state.get('raw_sources', [])}
    new_sources = [
        {**r, 'score': 7}
        for r in browser_results
        if r.get('url') and r.get('url') not in existing_urls
    ]

    state['browser_results'] = browser_results
    state['raw_sources']     = state.get('raw_sources', []) + new_sources
    state['status']          = 'browser_complete'

    return state
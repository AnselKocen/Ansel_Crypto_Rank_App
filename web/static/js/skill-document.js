/* A presentation-only reader. The exact source remains available below the document. */
(() => {
    'use strict';
    const textFor = (zh, en) => window.SiteLocale?.text(zh, en) ?? en;

    function scalar(value) {
        const text = value.trim();
        if (text.startsWith('"')) {
            const quoted = text.match(/^"((?:\\.|[^"\\])*)"/);
            if (quoted) {
                try { return JSON.parse(quoted[0]); }
                catch { return quoted[1].replace(/\\(["\\])/g, '$1'); }
            }
        }
        if (text.startsWith("'")) {
            const quoted = text.match(/^'((?:''|[^'])*)'/);
            if (quoted) return quoted[1].replace(/''/g, "'");
        }
        return text.replace(/[ \t]+#.*$/, '').trimEnd();
    }

    function dedent(lines, explicitIndent) {
        const nonempty = lines.filter(line => line.trim());
        const indent = explicitIndent ?? (nonempty.length ? Math.min(...nonempty.map(line => line.match(/^[ \t]*/)[0].length)) : 0);
        return lines.map(line => line.slice(Math.min(indent, line.length)));
    }

    function parseMetadata(lines) {
        const metadata = Object.create(null);
        for (let index = 0; index < lines.length;) {
            const match = lines[index].match(/^([^\s:#][^:]*):[ \t]*(.*)$/);
            if (!match) { index++; continue; }
            const key = scalar(match[1]);
            const first = match[2];
            const following = [];
            index++;
            while (index < lines.length && !/^[^\s:#][^:]*:/.test(lines[index])) following.push(lines[index++]);
            const block = first.match(/^([>|])([+-]?\d?|\d[+-]?)(?:[ \t]+#.*)?$/);
            const continuation = block ? following : following.filter(line => !/^#/.test(line));
            if (block) {
                const indicator = block[2].match(/[1-9]/);
                const content = dedent(following, indicator ? Number(indicator[0]) : undefined);
                let value = '';
                for (let row = 0; row < content.length; row++) {
                    const line = content[row], next = content[row + 1];
                    value += line;
                    if (row === content.length - 1) continue;
                    if (block[1] === '|') value += '\n';
                    else if (!line || /^[ \t]/.test(line) || (next && /^[ \t]/.test(next))) value += '\n';
                    else if (next) value += ' ';
                }
                if (block[2].includes('-')) value = value.replace(/\n+$/, '');
                else if (block[2].includes('+')) value += '\n';
                else value = value.replace(/\n+$/, '') + '\n';
                metadata[key] = value;
            } else if (/^["']/.test(first) && continuation.some(line => line.trim())) {
                metadata[key] = scalar([first, ...dedent(continuation)].join(' '));
            } else if (continuation.some(line => line.trim())) {
                metadata[key] = [scalar(first), ...dedent(continuation)].filter((line, row) => row || line).join('\n');
            } else metadata[key] = scalar(first);
        }
        return metadata;
    }

    function extractTitle(lines) {
        let fence = null;
        for (let index = 0; index < lines.length; index++) {
            const fenced = lines[index].match(/^ {0,3}(`{3,}|~{3,})/);
            if (fenced) {
                if (!fence) fence = fenced[1];
                else if (fenced[1][0] === fence[0] && fenced[1].length >= fence.length) fence = null;
                continue;
            }
            if (fence) continue;
            const heading = lines[index].match(/^ {0,3}#[ \t]+(.+)$/);
            const setext = lines[index].trim() && index + 1 < lines.length && /^ {0,3}=+[ \t]*$/.test(lines[index + 1]);
            if (!heading && !setext) continue;
            const title = (heading ? heading[1].replace(/[ \t]+#+[ \t]*$/, '') : lines[index].trim())
                .replace(/!?\[([^\]]+)\]\([^)]*\)/g, '$1').replace(/`+([^`]+)`+/g, '$1')
                .replace(/\*\*([^*]+)\*\*|__([^_]+)__/g, (_, bold, underlined) => bold || underlined).trim();
            lines.splice(index, heading ? 1 : 2);
            return title;
        }
        return '';
    }

    function prepare(raw, options = {}) {
        const original = String(raw ?? '');
        const lines = original.replace(/^\uFEFF/, '').replace(/\r\n?/g, '\n').split('\n');
        let metadata = Object.create(null);
        if (lines[0]?.trim() === '---') {
            const end = lines.findIndex((line, index) => index > 0 && /^(?:---|\.\.\.)[ \t]*$/.test(line));
            if (end > 0) {
                metadata = parseMetadata(lines.slice(1, end));
                lines.splice(0, end + 1);
            }
        }
        const heading = extractTitle(lines);
        const title = heading || options.title || metadata.title || metadata.name || options.name || textFor('技能文档', 'Skill document');
        const description = metadata.description || options.description || '';
        return {
            title,
            description,
            category: options.category == null ? '' : String(options.category),
            presentation: {
                title: options.title_zh ? {zh: options.title_zh, en: title} : null,
                description: options.description_zh ? {zh: options.description_zh, en: description} : null,
                category: options.category_zh ? {zh: options.category_zh, en: String(options.category || '')} : null
            },
            metadata,
            raw: original,
            body: lines.join('\n')
        };
    }

    const allowed = new Set('P H1 H2 H3 H4 H5 H6 UL OL LI BLOCKQUOTE PRE CODE STRONG EM DEL S HR BR TABLE CAPTION THEAD TBODY TFOOT TR TH TD A IMG'.split(' '));
    const discarded = new Set('SCRIPT STYLE IFRAME OBJECT EMBED LINK META BASE FORM INPUT BUTTON SELECT TEXTAREA SVG MATH TEMPLATE'.split(' '));

    function safeURL(value) {
        const href = value.trim();
        if (!href || /[\u0000-\u0020\u007f]/.test(href) || href.startsWith('//') || href.includes('\\')) return null;
        if (/^https?:\/\//i.test(href)) {
            try { const url = new URL(href); return /^(?:https?:)$/.test(url.protocol) ? href : null; }
            catch { return null; }
        }
        return /^[a-z][a-z\d+.-]*:/i.test(href) ? null : href;
    }

    function localReference(href) {
        return href && !/^(?:https?:\/\/|#)/i.test(href) && /(?:^|\/)references\/|\.md(?:[?#]|$)/i.test(href);
    }

    function cleanNodes(source, destination) {
        for (const node of source.childNodes) {
            if (node.nodeType === 3) { destination.appendChild(document.createTextNode(node.textContent)); continue; }
            const tag = (node.tagName || '').toUpperCase();
            if (node.nodeType === 1 && tag === 'INPUT' && node.getAttribute('type') === 'checkbox') {
                destination.appendChild(document.createTextNode(node.hasAttribute('checked') ? '[x] ' : '[ ] '));
                continue;
            }
            if (node.nodeType !== 1 || discarded.has(tag)) continue;
            if (!allowed.has(tag)) { cleanNodes(node, destination); continue; }
            const href = tag === 'A' ? node.getAttribute('href') || '' : '';
            if (localReference(href)) {
                const label = node.textContent;
                const code = document.createElement('code');
                code.textContent = href;
                if (label.trim() !== href) destination.appendChild(document.createTextNode(`${label} (`));
                destination.appendChild(code);
                if (label.trim() !== href) destination.appendChild(document.createTextNode(')'));
                continue;
            }
            const element = document.createElement(tag.toLowerCase());
            if (tag === 'A') {
                const safe = safeURL(href);
                if (safe) {
                    element.setAttribute('href', safe);
                    if (/^https?:\/\//i.test(safe)) {
                        element.target = '_blank';
                        element.rel = 'noopener noreferrer';
                    }
                }
                if (node.hasAttribute('title')) element.title = node.getAttribute('title');
            }
            if (tag === 'IMG') {
                const src = safeURL(node.getAttribute('src') || '');
                const alt = node.getAttribute('alt') || '';
                if (!src || localReference(src)) { destination.appendChild(document.createTextNode(alt)); continue; }
                element.src = src;
                element.alt = alt;
                element.loading = 'lazy';
                element.decoding = 'async';
            }
            for (const attribute of ['start', 'value', 'colspan', 'rowspan']) {
                if ((tag === 'OL' && attribute === 'start') || (tag === 'LI' && attribute === 'value') || (/^(?:TH|TD)$/.test(tag) && /^(?:colspan|rowspan)$/.test(attribute))) {
                    const value = node.getAttribute(attribute);
                    if (value && /^-?\d+$/.test(value)) element.setAttribute(attribute, value);
                }
            }
            if (/^(?:TH|TD)$/.test(tag) && /^(?:left|center|right)$/.test(node.getAttribute('align') || '')) element.setAttribute('align', node.getAttribute('align'));
            if (tag === 'CODE' && /^language-[\w+-]+$/.test(node.className)) element.className = node.className;
            cleanNodes(node, element);
            destination.appendChild(element);
        }
    }

    function render(model) {
        const article = document.createElement('article');
        article.className = 'skill-document';
        const header = document.createElement('header');
        header.className = 'skill-document-header';
        for (const [tag, className, value, key] of [
            ['p', 'skill-document-category', model.category, 'category'],
            ['h2', 'skill-document-title', model.title, 'title'],
            ['p', 'skill-document-description', model.description, 'description']
        ]) {
            const element = document.createElement(tag);
            element.className = className;
            const presentation = model.presentation?.[key];
            if (presentation) {
                element.dataset.zh = presentation.zh;
                element.dataset.en = presentation.en;
            }
            element.textContent = presentation ? textFor(presentation.zh, presentation.en) : value || '';
            if (!value) element.hidden = true;
            header.appendChild(element);
        }
        article.appendChild(header);
        const body = document.createElement('div');
        body.className = 'skill-document-body';
        try {
            if (typeof window.marked?.parse !== 'function') throw new Error('Markdown parser unavailable');
            const html = window.marked.parse(model.body, { async: false });
            if (typeof html !== 'string') throw new Error('Markdown parser is asynchronous');
            const template = document.createElement('template');
            template.innerHTML = html;
            cleanNodes(template.content, body);
        } catch {
            body.replaceChildren();
            const fallback = document.createElement('pre');
            fallback.textContent = model.body;
            body.appendChild(fallback);
        }
        for (const heading of body.querySelectorAll('h2')) {
            if (/\b(?:risks?|constraints?|limitations?|security|warnings?|cautions?)\b|风险|约束|限制|安全|警告/.test(heading.textContent.toLowerCase())) heading.classList.add('skill-document-important-heading');
        }
        for (const table of body.querySelectorAll('table')) {
            const wrapper = document.createElement('div');
            wrapper.className = 'skill-document-table';
            wrapper.tabIndex = 0;
            wrapper.dataset.ariaZh = '可横向滚动的表格';
            wrapper.dataset.ariaEn = 'Horizontally scrollable table';
            wrapper.setAttribute('aria-label', textFor('可横向滚动的表格', 'Horizontally scrollable table'));
            table.replaceWith(wrapper);
            wrapper.appendChild(table);
        }
        article.appendChild(body);
        const source = document.createElement('details');
        source.className = 'skill-document-source';
        const summary = document.createElement('summary');
        const summaryText = document.createElement('span');
        summaryText.dataset.zh = '文档源码与元数据';
        summaryText.dataset.en = 'Document source & metadata';
        summaryText.textContent = textFor(summaryText.dataset.zh, summaryText.dataset.en);
        summary.appendChild(summaryText);
        source.appendChild(summary);
        const metadata = document.createElement('dl');
        metadata.className = 'skill-document-metadata';
        for (const [key, value] of Object.entries(model.metadata || {})) {
            const term = document.createElement('dt'), definition = document.createElement('dd');
            term.textContent = key;
            definition.textContent = String(value);
            metadata.append(term, definition);
        }
        source.appendChild(metadata);
        const raw = document.createElement('pre'), code = document.createElement('code');
        raw.className = 'skill-document-raw';
        code.textContent = model.raw;
        raw.appendChild(code);
        source.appendChild(raw);
        article.appendChild(source);
        return article;
    }

    window.SkillDocument = { prepare, render };
})();

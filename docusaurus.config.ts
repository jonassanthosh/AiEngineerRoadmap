import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';
import remarkMath from 'remark-math';
import rehypeKatex from 'rehype-katex';

const config: Config = {
  title: 'AI Engineering Academy',
  tagline: 'From beginner to AI engineer in 6 months',
  favicon: 'img/favicon.svg',

  future: {
    v4: true,
  },

  url: 'https://your-domain.com',
  baseUrl: '/',

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',
  onBrokenAnchors: 'warn',

  stylesheets: [
    {
      href: 'https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap',
      type: 'text/css',
    },
    {
      href: 'https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css',
      type: 'text/css',
      integrity: 'sha384-nB0miv6/jRmo5YCBER1ssBmHIhKjhCR7yQqT2l5YMa3bWQ0RNnPBolI3yrLkFNQ',
      crossorigin: 'anonymous',
    },
  ],

  markdown: {
    format: 'detect',
  },

  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  themes: ['@docusaurus/theme-live-codeblock'],

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: 'curriculum',
          remarkPlugins: [remarkMath],
          rehypePlugins: [rehypeKatex],
        },
        blog: {
          showReadingTime: true,
          feedOptions: {
            type: ['rss', 'atom'],
            xslt: true,
          },
          onInlineTags: 'warn',
          onInlineAuthors: 'warn',
          onUntruncatedBlogPosts: 'warn',
        },
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    image: 'img/logo.svg',
    colorMode: {
      defaultMode: 'dark',
      respectPrefersColorScheme: true,
    },
    navbar: {
      title: 'AI Engineering Academy',
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'curriculumSidebar',
          position: 'left',
          label: 'Curriculum',
        },
        {to: '/blog', label: 'Blog', position: 'left'},
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Curriculum',
          items: [
            {
              label: 'Month 1 — Foundations',
              to: '/curriculum/month-1/what-is-ai',
            },
            {
              label: 'Month 4 — LLMs',
              to: '/curriculum/month-4/scaling-laws',
            },
            {
              label: 'Month 6 — Building Models',
              to: '/curriculum/month-6/reading-papers',
            },
          ],
        },
        {
          title: 'Community',
          items: [
            {
              label: 'Hugging Face',
              href: 'https://huggingface.co',
            },
            {
              label: 'arXiv',
              href: 'https://arxiv.org',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'Blog',
              to: '/blog',
            },
            {
              label: 'PyTorch',
              href: 'https://pytorch.org',
            },
            {
              label: 'Papers With Code',
              href: 'https://paperswithcode.com',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} AI Engineering Academy. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
      additionalLanguages: ['python', 'bash', 'json'],
    },
    liveCodeBlock: {
      playgroundPosition: 'bottom',
    },
  } satisfies Preset.ThemeConfig,
};

export default config;

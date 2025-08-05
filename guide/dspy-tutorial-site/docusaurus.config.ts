import {themes as prismThemes} from 'prism-react-renderer';
import type {Config} from '@docusaurus/types';
import type * as Preset from '@docusaurus/preset-classic';

// This runs in Node.js - Don't use client-side code here (browser APIs, JSX...)

const config: Config = {
  title: 'DSPy Zero-to-Hero Tutorial',
  tagline: 'Master DSPy: From Basic Prompting to Advanced Optimization',
  favicon: 'img/favicon.ico',

  // Future flags, see https://docusaurus.io/docs/api/docusaurus-config#future
  future: {
    v4: true, // Improve compatibility with the upcoming Docusaurus v4
  },

  // Set the production url of your site here
  url: 'https://vibe-school.github.io',
  // Set the /<baseUrl>/ pathname under which your site is served
  // For GitHub pages deployment, it is often '/<projectName>/'
  baseUrl: '/dspy/',

  // GitHub pages deployment config.
  // If you aren't using GitHub pages, you don't need these.
  organizationName: 'vibe-school', // Usually your GitHub org/user name.
  projectName: 'dspy', // Usually your repo name.

  onBrokenLinks: 'throw',
  onBrokenMarkdownLinks: 'warn',

  // Even if you don't use internationalization, you can use this field to set
  // useful metadata like html lang. For example, if your site is Chinese, you
  // may want to replace "en" with "zh-Hans".
  i18n: {
    defaultLocale: 'en',
    locales: ['en'],
  },

  presets: [
    [
      'classic',
      {
        docs: {
          sidebarPath: './sidebars.ts',
          routeBasePath: 'tutorial',
          // Please change this to your repo.
          // Remove this to remove the "edit this page" links.
          editUrl:
            'https://github.com/vibe-school/dspy/tree/main/guide/dspy-tutorial-site/',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      } satisfies Preset.Options,
    ],
  ],

  themeConfig: {
    // Replace with your project's social card
    image: 'img/docusaurus-social-card.jpg',
    navbar: {
      title: 'DSPy Tutorial',
      logo: {
        alt: 'DSPy Logo',
        src: 'img/logo.svg',
      },
      items: [
        {
          type: 'docSidebar',
          sidebarId: 'tutorialSidebar',
          position: 'left',
          label: 'Tutorial',
          to: '/tutorial/intro',
        },
        {
          href: 'https://github.com/stanfordnlp/dspy',
          label: 'DSPy GitHub',
          position: 'right',
        },
        {
          href: 'https://dspy.ai',
          label: 'DSPy Docs',
          position: 'right',
        },
      ],
    },
    footer: {
      style: 'dark',
      links: [
        {
          title: 'Docs',
          items: [
            {
              label: 'Start Tutorial',
              to: '/tutorial/intro',
            },
          ],
        },
        {
          title: 'Resources',
          items: [
            {
              label: 'DSPy Official Docs',
              href: 'https://dspy.ai',
            },
            {
              label: 'DSPy GitHub',
              href: 'https://github.com/stanfordnlp/dspy',
            },
            {
              label: 'YouTube Course',
              href: 'https://youtu.be/5Bym0ffALaU',
            },
          ],
        },
        {
          title: 'More',
          items: [
            {
              label: 'Source Code',
              href: 'https://github.com/vibe-school/dspy',
            },
            {
              label: 'AVB Patreon',
              href: 'https://www.patreon.com/NeuralBreakdownwithAVB',
            },
          ],
        },
      ],
      copyright: `Copyright © ${new Date().getFullYear()} DSPy Tutorial. Built with Docusaurus.`,
    },
    prism: {
      theme: prismThemes.github,
      darkTheme: prismThemes.dracula,
    },
  } satisfies Preset.ThemeConfig,
};

export default config;

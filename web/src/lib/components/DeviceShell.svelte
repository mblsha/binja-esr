<script lang="ts">
	import type { RomModel } from '../rom_model';
	import type { InputContact } from '../emulator/host_inputs';
	import { deviceLayout, rectStyle } from '../device_layout';
	import VirtualKeyboard from './VirtualKeyboard.svelte';
	import type { HostKeyboardMode } from '../keymap';

	export let model: RomModel;
	export let disabled = false;
	export let hostKeyboardMode: HostKeyboardMode = 'symbols';
	export let physicalHighlights = new Set<InputContact>();
	export let deliveredContacts = new Set<InputContact>();
	export let onPress: (code: InputContact, owner: string) => void;
	export let onRelease: (code: InputContact, owner: string, cancel: boolean) => void;
	export let onCancelAll: () => void;
	$: layout = deviceLayout(model);
</script>

<section class="bench" aria-label={`${model.toUpperCase()} device view`}>
	<div class="bench-caption">
		<span>DEVICE VIEW / {model.toUpperCase()}</span><span>REFERENCE-BASED · NOT SCAN-DERIVED</span>
	</div>
	<!-- svelte-ignore a11y_no_noninteractive_tabindex (The scroll region needs focus for keyboard panning.) -->
	<div
		class="pan"
		data-host-scroll
		tabindex="0"
		role="region"
		aria-label="Device layout; horizontally scrollable on small screens"
	>
		<div
			class={`device ${model}`}
			data-testid="device-shell"
			data-layout-version={layout.version}
			style={`aspect-ratio:${layout.width}/${layout.height};`}
		>
			{#if model === 'iq-7000'}
				<div class="left-case" aria-hidden="true"></div>
				<div class="right-case" aria-hidden="true"></div>
				<div class="hinge" aria-hidden="true"></div>
				<div class="brand iq-brand" aria-hidden="true">SHARP <small>IQ-7000</small></div>
				<div class="iq-glass" aria-hidden="true"></div>
				<div class="script-brand" aria-hidden="true">Electronic<br />Organizer</div>
				<div
					class="card-bay"
					role="img"
					aria-label="Replaceable card area; decorative artwork placeholder, not a second LCD"
				>
					<span>IC CARD</span>
					<div class="card-rule"></div>
					<strong>REPLACEABLE<br />CARD AREA</strong>
					<small>Artwork placeholder<br />Not a second display</small>
				</div>
				<div class="latch" aria-hidden="true">LOCK<br /><i></i><br />RELEASE<br /><br />EJECT</div>
			{:else}
				<div class="pc-rail" aria-hidden="true"></div>
				<div class="brand pc-brand" aria-hidden="true">SHARP <small>POCKET COMPUTER&nbsp; PC-E500</small></div>
				<div class="engineer" aria-hidden="true">
					ENGINEER SOFTWARE<br /><small>SCIENTIFIC CONSTANTS AND FORMULAS</small>
				</div>
				<div class="pc-glass" aria-hidden="true"></div>
				<div class="key-divider" aria-hidden="true"></div>
			{/if}
			<div class="lcd-window" style={rectStyle(layout.lcd, layout)}><slot /></div>
			<VirtualKeyboard
				{model}
				{layout}
				{disabled}
				{hostKeyboardMode}
				{physicalHighlights}
				{deliveredContacts}
				{onPress}
				{onRelease}
				{onCancelAll}
			/>
		</div>
	</div>
	<div class="bench-foot">
		<span>Actual emulated LCD · Physical key contacts</span><span
			><i></i> Dotted keys are not mapped. Smaller screen? Pan the case.</span
		>
	</div>
</section>

<style>
	.bench {
		border: 1px solid #35434a;
		border-radius: 16px;
		background: radial-gradient(ellipse at 50% 25%, #3a4b51, #202c32 70%);
		padding: 19px 22px 16px;
		box-shadow: inset 0 1px #ffffff0a;
	}
	.bench-caption,
	.bench-foot {
		display: flex;
		justify-content: space-between;
		gap: 10px;
		flex-wrap: wrap;
		color: #b8c6ca;
		font:
			10px/1.5 ui-monospace,
			monospace;
		letter-spacing: 1px;
	}
	.bench-caption {
		margin-bottom: 26px;
	}
	.bench-foot {
		margin-top: 25px;
		letter-spacing: 0;
		color: #acb9bd;
	}
	.bench-foot i {
		display: inline-block;
		width: 5px;
		height: 5px;
		border-radius: 50%;
		background: #c89260;
		margin-right: 5px;
	}
	.pan {
		overflow-x: auto;
		padding: 3px 4px 20px;
		scrollbar-color: #829391 #243139;
	}
	.pan:focus-visible {
		outline: 2px solid #91d4d0;
		outline-offset: 3px;
		border-radius: 6px;
	}
	.device {
		container-type: inline-size;
		position: relative;
		margin: 0 auto;
		color: #efeee2;
		user-select: none;
	}
	.pc-e500 {
		min-width: 760px;
		max-width: 1080px;
		width: 100%;
		border-radius: 16px 16px 24px 24px;
		background: linear-gradient(100deg, #66665f, #4c4c47 49%, #5d5d56);
		box-shadow:
			inset 0 2px 3px #babbb1,
			inset 0 -5px 7px #161c1a,
			0 18px 20px #10181b90;
		border: 3px solid #303835;
	}
	.iq-7000 {
		width: 78%;
		max-width: 850px;
		min-width: 660px;
		filter: drop-shadow(0 15px 9px #10181b88);
	}
	.left-case,
	.right-case {
		position: absolute;
		top: 0;
		bottom: 0;
		border: 3px solid #151d21;
		background: linear-gradient(120deg, #3a4245, #1c2529 65%, #303b40);
		box-shadow:
			inset 0 2px 4px #8e999a,
			inset 0 -4px 5px #0b1216;
		border-radius: 25px 16px 19px 27px;
	}
	.left-case {
		left: 0;
		width: 48.8%;
	}
	.right-case {
		left: 50.9%;
		right: 0;
		border-radius: 17px 28px 28px 18px;
		background: linear-gradient(105deg, #666b6b, #484f52 60%, #606669);
	}
	.hinge {
		position: absolute;
		top: 2%;
		bottom: 2%;
		left: 49%;
		width: 2.7%;
		background: repeating-linear-gradient(0deg, #222e33 0%, #505e65 14%, #141c20 15%, #141c20 18%);
		border-radius: 7px;
		box-shadow: inset 2px 0 4px #879496;
	}
	.brand {
		position: absolute;
		font:
			900 2.6cqw/1 Arial,
			sans-serif;
		letter-spacing: -1px;
	}
	.brand small {
		font:
			500 1.2cqw/1 Arial,
			sans-serif;
		letter-spacing: 0;
		margin-left: 12px;
	}
	.pc-brand {
		top: 5%;
		left: 4%;
	}
	.pc-rail {
		position: absolute;
		top: 12%;
		left: 1.2%;
		right: 1.2%;
		height: 3.1%;
		background: linear-gradient(#919085, #585b54, #333e39);
		border-radius: 7px;
		box-shadow: 0 2px 3px #262a26;
	}
	.engineer {
		position: absolute;
		top: 4%;
		right: 3.4%;
		color: #cabb86;
		font:
			500 1.5cqw/1 Arial,
			sans-serif;
	}
	.engineer small {
		font-size: 0.92cqw;
	}
	.pc-glass {
		position: absolute;
		left: 3.5%;
		top: 18%;
		width: 60%;
		height: 28%;
		background: linear-gradient(130deg, #192322, #0b1311);
		border: 2px solid #a8aaa066;
		box-shadow: inset 0 0 5px #000;
		border-radius: 7px;
	}
	.key-divider {
		position: absolute;
		top: 49%;
		left: 66%;
		right: 2.5%;
		height: 1px;
		background: #a3a59755;
	}
	.iq-brand {
		left: 5.8%;
		top: 4.5%;
		color: #dccb9b;
		font-size: 2.2cqw;
	}
	.iq-brand small {
		font-size: 1.1cqw;
	}
	.iq-glass {
		position: absolute;
		left: 4.2%;
		top: 10.8%;
		width: 42%;
		height: 32%;
		border-radius: 5px;
		border: 2px solid #ad9560;
		background: #131f20;
		box-shadow: 0 0 0 9px #131d22;
	}
	.script-brand {
		position: absolute;
		top: 43%;
		left: 30%;
		color: #e2cf9e;
		font:
			italic 2.5cqw/0.83 Georgia,
			serif;
		transform: rotate(-8deg);
	}
	.card-bay {
		position: absolute;
		left: 10.6%;
		top: 53%;
		width: 28.3%;
		height: 34%;
		border: 2px solid #9caaac;
		border-radius: 5px;
		background: linear-gradient(135deg, #8d938e, #646e6c);
		box-shadow:
			0 0 0 11px #131d21,
			inset 0 0 12px #1d262766;
		color: #e6e5ce;
		display: flex;
		flex-direction: column;
		justify-content: center;
		align-items: center;
		gap: 2cqw;
		text-align: center;
		font:
			500 1.2cqw/1.4 Arial,
			sans-serif;
		letter-spacing: 1px;
	}
	.card-bay strong {
		font-size: 1.6cqw;
		font-weight: 400;
	}
	.card-bay small {
		font-size: 0.95cqw;
		color: #e5e7de;
	}
	.card-rule {
		height: 1px;
		width: 70%;
		background: #dcd7bc88;
	}
	.latch {
		position: absolute;
		top: 61%;
		left: 3.5%;
		color: #c8b580;
		font:
			0.65cqw/1.2 Arial,
			sans-serif;
	}
	.latch i {
		display: block;
		margin-top: 1cqw;
		width: 1cqw;
		height: 3.2cqw;
		border-radius: 3px;
		background: #899299;
		box-shadow: inset 0 0 3px #131c21;
	}
	.lcd-window {
		position: absolute;
		display: flex;
		align-items: center;
		justify-content: center;
		overflow: hidden;
		background: #bfc0b3;
		box-shadow: inset 0 0 5px #18231e;
		border-radius: 2px;
	}
	.lcd-window :global(.lcd-display) {
		width: 100%;
		max-height: 100%;
	}
	@media (max-width: 700px) {
		.bench {
			padding: 14px 10px 12px;
		}
		.bench-caption {
			margin-bottom: 17px;
		}
	}
</style>
